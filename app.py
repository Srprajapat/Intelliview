from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import cv2
import numpy as np
import base64
import os
import uuid
import operator
from typing import Literal, List, Annotated, Optional
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from document_extractor import extract_text_auto  
import warnings
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import traceback
import json
from ultralytics import YOLO
import logging
import time
from gaze_tracker import GazeTracker

logging.basicConfig(filename='debug.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Robust LLM call wrapper with exponential backoff ---
def safe_llm_invoke(llm, messages, max_attempts=5, structured=None):
    """Wraps LLM calls with exponential backoff to handle 400/503 from Groq."""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            if structured:
                result = llm.with_structured_output(structured).invoke(messages)
            else:
                result = llm.invoke(messages)
            return result
        except Exception as e:
            last_error = e
            err_str = str(e)
            logging.warning(f"LLM call attempt {attempt}/{max_attempts} failed: {err_str[:200]}")
            # Exponential backoff: 2s, 4s, 8s, 16s, 32s
            wait = min(2 ** attempt, 32)
            if '503' in err_str or 'overloaded' in err_str.lower() or 'rate' in err_str.lower():
                logging.info(f"Rate limited / overloaded, waiting {wait}s before retry...")
                time.sleep(wait)
            elif '400' in err_str:
                logging.info(f"Bad request, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                time.sleep(wait)
    raise last_error

def truncate_text(text, max_chars=4000):
    """Truncate text to fit within token limits. ~4 chars per token, 4000 chars ≈ 1000 tokens."""
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated for brevity]"

def to_serializable(obj):
    if obj is None or isinstance(obj, (str, int, float, bool)): return obj
    try:
        from pydantic import BaseModel as _BaseModel
        if isinstance(obj, _BaseModel): return to_serializable(obj.model_dump())
    except Exception: pass
    if isinstance(obj, dict): return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [to_serializable(v) for v in obj]
    if hasattr(obj, 'model_dump'):
        try: return to_serializable(obj.model_dump())
        except Exception: pass
    if hasattr(obj, 'dict'):
        try: return to_serializable(obj.dict())
        except Exception: pass
    try: return str(obj)
    except Exception: return None

def save_result_to_file(result_dict, filename=None):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    if filename is None: filename = f"result_{uuid.uuid4().hex}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False)
        return filename
    except Exception:
        traceback.print_exc()
        return None

def save_extracted_text(text, kind: str):
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    fname = f"{kind}_{uuid.uuid4().hex}.txt"
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    try:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(text or "")
        return fname
    except Exception:
        traceback.print_exc()
        return None

def load_extracted_text_from_file(field_name):
    fn = session.get(field_name)
    if not fn: return None
    path = os.path.join(app.config['UPLOAD_FOLDER'], fn)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception:
        traceback.print_exc()
        return None

def load_result_from_session():
    rf = session.get('result_file')
    if rf:
        path = os.path.join(app.config['UPLOAD_FOLDER'], rf)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            traceback.print_exc()
            return {}
    return session.get('result', {})

def compute_integrity(base_integrity: float, integrity_deducted) -> float:
    try: deducted = float(integrity_deducted or 0)
    except Exception: deducted = 0.0
    deduction = deducted * 2.0
    try: base = float(base_integrity or 100.0)
    except Exception: base = 100.0
    return max(0.0, base - deduction)

warnings.filterwarnings('ignore', category=UserWarning, module='face_recognition')
load_dotenv()

# ==========================================
# Specialized LLM Models for Different Tasks
# ==========================================
llm_jd = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", max_retries=3, temperature=0.2)

llm_resume = ChatGroq(model="moonshotai/kimi-k2-instruct-0905", max_retries=3, temperature=0.1)

llm_mcq = ChatGroq(model="llama-3.1-8b-instant", max_retries=3, temperature=0.7)

llm_speech = ChatGroq(model="llama-3.1-8b-instant", max_retries=3, temperature=0.7)


app = Flask(__name__)
app.secret_key = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'tempDir'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
proctor_processors = {}

class ProctoringProcessor:
    def __init__(self):
        self.frame_count = 0
        self.integrity_deducted = 0
        self.gaze_away = 0
        try:
            self.yolo_model = YOLO('yolov8n.pt') 
            self.yolo_available = True
        except Exception as e:
            logging.error(f"YOLO loading failed: {e}")
            self.yolo_available = False
        self.model_points = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
            (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1)) 
        # gaze tracker instance
        self.gaze_tracker = GazeTracker()

    def detect_objects(self, img):
        prohibited_objects = []
        if not self.yolo_available: return prohibited_objects
        try:
            results = self.yolo_model(img, conf=0.5, verbose=False) 
            class_names = self.yolo_model.names
            prohibited_items = ['cell phone', 'smartphone', 'mobile', 'phone', 'laptop', 'notebook', 'computer', 'book']
            for result in results:
                for box in result.boxes:
                    class_name = class_names[int(box.cls[0])]
                    if any(item in class_name.lower() for item in prohibited_items):
                        prohibited_objects.append(class_name)
        except Exception: pass
        return prohibited_objects

    def estimate_head_pose(self, landmarks, img_shape):
        try:
            if self.camera_matrix is None:
                h, w = img_shape[:2]
                self.camera_matrix = np.array([[w, 0, w/2], [0, w, h/2], [0, 0, 1]], dtype="double")
            image_points = np.array([
                landmarks["nose_bridge"][3], landmarks["chin"][8], landmarks["left_eye"][0],
                landmarks["right_eye"][3], landmarks["top_lip"][0], landmarks["bottom_lip"][4]
            ], dtype="double")
            success, rotation_vector, translation_vector = cv2.solvePnP(self.model_points, image_points, self.camera_matrix, self.dist_coeffs)
            if success:
                rmat, _ = cv2.Rodrigues(rotation_vector)
                angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
                yaw = angles[1] * 180.0 / np.pi
                pitch = angles[0] * 180.0 / np.pi
                if abs(yaw) > 180 or abs(pitch) > 180: return None, None
                return yaw, pitch
            return None, None
        except Exception: return None, None

    def process_frame(self, frame_data):
        self.frame_count += 1
        if self.frame_count % 3 != 0: return frame_data, self.integrity_deducted, self.gaze_away
        
        image_bytes = base64.b64decode(frame_data.split(',')[1])
        img = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Check image brightness
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        logging.debug(f"Image brightness: {brightness:.2f}")
        if brightness < 50:  # Too dark
            logging.warning("Image too dark for face detection")

        # Check image blur
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        logging.debug(f"Image blur (Laplacian variance): {laplacian_var:.2f}")
        if laplacian_var < 100:  # Blurry
            logging.warning("Image too blurry for face detection")

        prohibited_objects = self.detect_objects(img)
        if prohibited_objects:
            self.integrity_deducted += 3 
            logging.info(f"Integrity deducted 3 for prohibited objects: {prohibited_objects}")

        try:

            gaze_info, annotated_img = self.gaze_tracker.process_frame(img)

            img = annotated_img

            faces = gaze_info.get("faces",0)

            if faces == 0:
                self.integrity_deducted += 2
                logging.info("Integrity deducted 2 due to no face detected")

            elif faces > 1:
                self.integrity_deducted += 2
                logging.info("Integrity deducted 2 due to multiple faces detected")

            direction = gaze_info.get("gaze")
            violations = gaze_info.get("violations", [])

            if direction and direction != "center":
                self.gaze_away += 1
                logging.info(f"Gaze away detected ({direction})")

            if violations:
                penalty = len(violations) * 3
                self.integrity_deducted += penalty
                logging.info(
                    f"Integrity deducted {penalty} for gaze violations {violations}"
                )

        except Exception as e:
            logging.error(e)

        return frame_data, self.integrity_deducted, self.gaze_away

# --- LangGraph Pydantic Models ---
class requiredSkills(BaseModel):
    mustHave : List[str]
    goodToHave : List[str]

class claimedSkill(BaseModel):
    skill : str
    found : bool
    context : Annotated[str,"line in a resume where candidate is claiming to have this skill"]

class r_question(BaseModel):
    topic : str = Field(description = "Topic on which question is generated")
    question : str = Field(description="Question generated based on topics")
    correct_answer : str = Field(description="Correct answer of the question")
    options : List[str] = Field(description="Options for the question")
    level : Literal['easy','medium','hard']

class MCQ(BaseModel):
    question: str = Field(description="Question generated by bridging context and topic")
    
    # Force the LLM to write out the full text
    options: List[str] = Field(
        description="List of exactly 4 fully written answer choices. Example: ['Full text of option 1', 'Full text of option 2', 'Full text of option 3', 'I do not know/None of the above']. Do NOT output single letters."
    )
    
    # Replaced 'a,b,c,d' with the exact string match
    correct_answer: str = Field(description="The exact text of the correct option from the options list")
    
    topic: str = Field(description="Topic on which question is generated")
    question_level: Literal['surface','deep level 1', 'deep level 2']

class v_question(BaseModel):
    surface_level : MCQ = Field(description="Surface level question generated based on topic")
    deep_level_1 : MCQ = Field(description="Deep level 1 question generated based on topic")
    deep_level_2 : MCQ = Field(description="Deep level 2 question generated based on topic")

class customState(BaseModel):
    resume :  Optional[str] = Field(default=None)
    jd : Optional[str] = Field(default = None)
    reqSkills : Optional[requiredSkills] = Field(default=None)
    exp: Optional[Literal["Fresher", "Junior", "Senior"]] = Field(default=None)
    tech_stack: str = Field(default="")
    easy_question : int = Field(default=4)
    medium_question : int = Field(default=3)
    hard_question : int = Field(default=3)
    claimedSkills: List[claimedSkill] = Field(default_factory=list)
    projects: List[str] = Field(default_factory=list)
    achievements: List[str] = Field(default_factory=list)
    verification_topics: List[str] = Field(default_factory=list)
    jd_topics : List[str] = Field(default_factory=list)
    gap_analysis_topics: List[str] = Field(default_factory=list)
    current_topic: Optional[str] = Field(default=None)
    current_context : str = Field(default="")
    current_level : Literal["surface","deep level 1", "deep level 2"] = Field(default="surface")
    current_stage : Literal["required_skills", "verification"] = Field(default="required_skills")
    easy_questions_asked : int = Field(default=0)
    medium_questions_asked : int = Field(default=0)
    hard_questions_asked : int = Field(default=0)
    topic_covered : List[str] = Field(default=[])
    max_score : Annotated[float,operator.add] = Field(default=0)
    score: Annotated[float,operator.add] = Field(default=0)
    current_bonus : float = Field(default=0)
    integrity: Annotated[float,operator.add] = Field(default=100.0)
    candidate_name: str = Field(default="Candidate")
    message_history : Annotated[List[BaseMessage], operator.add] = Field(default_factory=list)
    choosed_ans: Optional[str] = Field(default=None)
    questions_asked: int = Field(default=0)
    max_questions: int = Field(default=10)
    required_question : List[r_question] = Field(default_factory=list)
    verification_question : List[v_question] = Field(default_factory=list)

class ResumeAnalysisState(BaseModel):
    candidate_name: str = Field(description="Full name of the candidate found in the resume header")
    claimedSkills : Annotated[List[claimedSkill],"Checking all the skill from required skill"]
    projects : Annotated[List[str], "projects that candidate mentioned in the resume"]
    achievements : Annotated[List[str], "achievements that candidate mentioned in the resume"]

# --- LangGraph Nodes ---
def p1s1(state)->dict:
    state = state.model_dump()
    parser = JsonOutputParser()
    system_prompt = """You are an expert Technical Recruiter. Analyze a Job Description (JD).
    Output ONLY valid JSON:
    { "reqSkills": { "mustHave": ["skill1"], "goodToHave": ["skill3"] }, "exp": "Fresher|Junior|Senior", "tech_stack": "string" }"""
    jd_text = truncate_text(state['jd'], 3500)
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=jd_text)]
    response = safe_llm_invoke(llm_jd, messages)
    return parser.parse(response.content)

def p1s2(state) -> dict:
    state = state.model_dump()
    parser = PydanticOutputParser(pydantic_object=ResumeAnalysisState)
    target_skills = state['reqSkills']['mustHave'] + state['reqSkills']['goodToHave']
    resume_text = truncate_text(state['resume'], 3000)
    system_prompt = f"Expert Resume Auditor. Verify claims against target skills. Output ONLY valid JSON matching this schema: candidate_name (str), claimedSkills (list of skill/found/context), projects (list of str), achievements (list of str)."
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"TARGET SKILLS: {target_skills}\nRESUME: {resume_text}")]
    response = safe_llm_invoke(llm_resume, messages)
    return parser.parse(response.content).model_dump()

def p1s3(state) -> dict:
    state = state.model_dump()
    req_must = state['reqSkills']['mustHave']
    all_req_set = set(skill.lower() for skill in req_must + state['reqSkills']['goodToHave'])
    claimed_set = {item['skill'].lower() for item in state['claimedSkills'] if item['found']}
    def priority(t): return 0 if t in req_must else 1
    return {
        "verification_topics": sorted(list(all_req_set.intersection(claimed_set)), key=priority),
        "gap_analysis_topics": sorted(list(all_req_set.difference(claimed_set)), key=priority),
        "jd_topics": sorted(list(all_req_set), key=priority)
    }

class required_question(BaseModel):
    questions : List[r_question]

def r_p2s1(state) -> dict:
    state = state.model_dump()
    topics = state['jd_topics'][:8]  # Limit topics to avoid token overflow
    system_prompt = "Generate conceptual interview MCQ questions for the given topics. For each question provide: topic (str), question (str), correct_answer (str), options (list of 4 full-text strings), level (easy/medium/hard). Return JSON with key 'questions' containing the list."
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Topics: {topics}")]
    time.sleep(1)  # Small delay to avoid rate limiting after previous calls
    parsed_obj = safe_llm_invoke(llm_mcq, messages, structured=required_question)
    return {"required_question": parsed_obj.questions, "max_score": len(parsed_obj.questions) * 10}

def v_p2s1(state) -> dict:
    """Generate verification questions using plain JSON (not tool calling) to avoid tool_use_failed errors."""
    import re as _re
    state = state.model_dump()
    topics = state['verification_topics'] if state['verification_topics'] else state['jd_topics']
    topics = topics[:5]  # Limit topics to avoid token overflow
    system_prompt = """Generate verification MCQ questions. Return ONLY valid JSON (no markdown, no explanation).
Format:
{"questions": [
  {
    "surface_level": {"question": "...", "options": ["A text", "B text", "C text", "D text"], "correct_answer": "exact text of correct option", "topic": "...", "question_level": "surface"},
    "deep_level_1": {"question": "...", "options": ["A text", "B text", "C text", "D text"], "correct_answer": "exact text of correct option", "topic": "...", "question_level": "deep level 1"},
    "deep_level_2": {"question": "...", "options": ["A text", "B text", "C text", "D text"], "correct_answer": "exact text of correct option", "topic": "...", "question_level": "deep level 2"}
  }
]}
Rules: Each topic = 1 entry with 3 levels. Options = 4 full sentences. correct_answer must exactly match one option."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=f"Topics: {topics}")]
    time.sleep(2)  # Delay to avoid rate limiting

    last_error = None
    for attempt in range(1, 4):
        try:
            response = safe_llm_invoke(llm_speech, messages)  # Plain text, NO structured output
            content = response.content.strip()
            logging.debug(f"v_p2s1 raw response (attempt {attempt}): {content[:500]}")
            # Extract JSON object from response (skip any markdown fences or text)
            json_match = _re.search(r'\{[\s\S]*\}', content)
            if not json_match:
                raise ValueError("No JSON object found in LLM response")
            parsed = json.loads(json_match.group())
            raw_questions = parsed.get('questions', [])
            if not raw_questions:
                raise ValueError("Empty questions list in parsed JSON")
            v_questions = []
            for q in raw_questions:
                v_q = v_question(
                    surface_level=MCQ(**q['surface_level']),
                    deep_level_1=MCQ(**q['deep_level_1']),
                    deep_level_2=MCQ(**q['deep_level_2'])
                )
                v_questions.append(v_q)
            logging.info(f"v_p2s1 SUCCESS: Generated {len(v_questions)} verification question sets")
            return {"verification_question": v_questions, "max_score": len(v_questions) * 10}
        except Exception as e:
            last_error = e
            logging.warning(f"v_p2s1 JSON parse attempt {attempt}/3 failed: {e}")
            time.sleep(2)
    # If all parse attempts fail, return empty so interview can still proceed with required questions only
    logging.error(f"v_p2s1 FAILED all attempts: {last_error}")
    return {"verification_question": [], "max_score": 0}

def build_graph1():
    graph1 = StateGraph(customState)
    graph1.add_node(p1s1, "p1s1")
    graph1.add_node(p1s2, "p1s2")
    graph1.add_node(p1s3, "p1s3")
    graph1.add_node(r_p2s1, "r_p2s1")
    graph1.add_node(v_p2s1, "v_p2s1")
    graph1.add_edge(START, "p1s1")
    graph1.add_edge("p1s1", "p1s2")
    graph1.add_edge("p1s2", "p1s3")
    graph1.add_edge("p1s3", "r_p2s1")
    graph1.add_edge("p1s3", "v_p2s1")
    graph1.add_edge("v_p2s1", END)
    return graph1.compile()


# --- Flask Routes ---
@app.route('/')
def index():
    if 'sid' not in session: session['sid'] = uuid.uuid4().hex
    if 'page' not in session: session['page'] = 'upload'
    if session.get('page') == 'result':
        state = load_result_from_session()
        return render_template('index.html', state=state)
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        jd_file = request.files.get('job_description')
        resume_file = request.files.get('resume')
        if jd_file and jd_file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(jd_file.filename))
            jd_file.save(filepath)
            ex = extract_text_auto(filepath)
            if ex['text']:
                session['job_description_file'] = save_extracted_text(ex['text'], 'jd')
                flash('Job Description uploaded successfully!', 'success')
        if resume_file and resume_file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume_file.filename))
            resume_file.save(filepath)
            ex = extract_text_auto(filepath)
            if ex['text']:
                session['resume_file'] = save_extracted_text(ex['text'], 'resume')
                flash('Resume uploaded successfully!', 'success')
                session['page'] = 'preface'
                return redirect(url_for('preface'))
    session['page'] = 'upload'
    return render_template('index.html')

@app.route('/preface')
def preface():
    if 'job_description_file' not in session or 'resume_file' not in session: return redirect(url_for('upload'))
    if 'analysis_text' not in session:
        try:
            jd_text = truncate_text(load_extracted_text_from_file('job_description_file') or '', 2000)
            resume_text = truncate_text(load_extracted_text_from_file('resume_file') or '', 2000)
            prompt = f"Analyze JD and Resume briefly. Output markdown sections: ### Core Job Requirements, ### Your Profile Highlights, ### Interview Focus Areas.\nJD:{jd_text}\nResume:{resume_text}"
            response = safe_llm_invoke(llm_jd, [HumanMessage(content=prompt)])
            session['analysis_text'] = response.content
        except Exception as e:
            logging.error(f"Preface analysis failed: {e}")
            session['analysis_text'] = "### Analysis Unavailable\nPlease proceed to start the interview."
    session['page'] = 'preface'
    return render_template('index.html')

@app.route('/enable_camera', methods=['POST'])
def enable_camera():
    session['camera_enabled'] = True
    return jsonify({'status': 'success'})

@app.route('/start_interview', methods=['POST'])
def start_interview():
    if not session.get('questions_prepared', False):
        try:
            app_graph = build_graph1()
            initial_state = {
                'resume': load_extracted_text_from_file('resume_file'),
                'jd': load_extracted_text_from_file('job_description_file'),
                'integrity': 100.0, 'max_questions': 12, 'questions_asked': 0, 'topic_covered': []
            }
            result_obj = app_graph.invoke(initial_state)
            session['result_file'] = save_result_to_file(to_serializable(result_obj))
            session['questions_prepared'] = True
        except Exception as e:
            logging.error(f"Question generation failed: {traceback.format_exc()}")
            error_msg = f'Failed to prepare questions: {str(e)[:100]}. Please try again.'
            # Support both AJAX and normal form POST
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify({'success': False, 'error': error_msg}), 500
            flash(error_msg, 'error')
            return redirect(url_for('preface'))
    session['page'] = 'interview'
    session['interview_started'] = True
    session['integrity_deducted'] = 0
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return jsonify({'success': True, 'redirect': url_for('interview')})
    return redirect(url_for('interview'))

@app.route('/interview', methods=['GET', 'POST'])
def interview(): return render_template('index.html')

@app.route('/speech_interview')
def speech_interview():
    if not session.get('speech_phase_started', False): return redirect(url_for('interview'))
    session['page'] = 'speech_interview'
    return render_template('index.html')

@app.route('/get_speech_question')
def get_speech_question():
    result = load_result_from_session()
    asked = int(session.get('speech_questions_asked', 0))
    max_speech = 3 
    if asked >= max_speech: return jsonify({'question': None})
    all_q = (result.get('required_question') or []) + (result.get('verification_question') or [])
    if asked < len(all_q):
        q_obj = all_q[asked]
        if 'surface_level' in q_obj: q_obj = q_obj['surface_level']
        return jsonify({'question': q_obj.get('question', ''), 'correct_answer': q_obj.get('correct_answer', ''), 'current_question': asked + 1, 'total_questions': max_speech})
    return jsonify({'question': None})

# NEW TEXT-ONLY SUBMISSION ROUTE (Browser Native STT handles the audio)
@app.route('/submit_speech_answer', methods=['POST'])
def submit_speech_answer():
    try:
        data = request.get_json()
        text_answer = data.get('answer_text', '').strip()
        
        if not text_answer:
            return jsonify({
                'recognized_text': '', 'correct': False, 'empty_audio': True,
                'message': "I didn't quite catch that. Could you please repeat your answer?"
            })

        result = load_result_from_session()
        asked = int(session.get('speech_questions_asked', 0))
        all_q = (result.get('required_question') or []) + (result.get('verification_question') or [])
        correct = False
        
        if asked < len(all_q):
            q_obj = all_q[asked]
            if 'surface_level' in q_obj: q_obj = q_obj['surface_level']
            correct_answer = q_obj.get('correct_answer', '').lower()
            user_answer = text_answer.lower()
            correct = correct_answer in user_answer or any(word in user_answer for word in correct_answer.split() if len(word) > 3)
        
        if correct: session['speech_score'] = session.get('speech_score', 0) + 10
        session['speech_questions_asked'] = asked + 1

        return jsonify({'recognized_text': text_answer, 'correct': correct, 'empty_audio': False, 'score': session.get('speech_score', 0)})
    except Exception as e: return jsonify({'error': 'Failed processing'}), 500

@app.route('/end_interview', methods=['POST'])
def end_interview():
    if not session.get('speech_phase_started', False):
        session['speech_phase_started'] = True
        session['speech_questions_asked'] = 0
        session['speech_score'] = 0
        session['page'] = 'speech_interview'
        return redirect(url_for('speech_interview'))
    session['interview_ended'] = True
    session['page'] = 'result'
    if session.get('sid') in proctor_processors: proctor_processors.pop(session.get('sid'), None)
    return redirect(url_for('result'))

@app.route('/result')
def result():
    if not session.get('interview_ended', False): return redirect(url_for('interview'))
    state = load_result_from_session()
    integrity_deducted = session.get('integrity_deducted', 0)
    gaze_away = session.get('gaze_away', 0)
    state['integrity_deducted'] = integrity_deducted
    state['gaze_away'] = gaze_away
    state['integrity'] = compute_integrity(state.get('integrity', 100.0), integrity_deducted)
    
    # Calculate actual scores based on questions answered
    req_list = state.get('required_question', []) or []
    ver_list = state.get('verification_question', []) or []
    total_mcq_questions = len(req_list) + len(ver_list)
    mcq_raw = state.get('score', 0)
    speech_raw = session.get('speech_score', 0)
    speech_asked = int(session.get('speech_questions_asked', 0))
    
    # Normalize to percentage out of 100
    mcq_max = max(total_mcq_questions * 10, 1)  # avoid division by zero
    speech_max = max(speech_asked * 10, 1)
    state['mcq_score'] = round((mcq_raw / mcq_max) * 50, 1)  # MCQ worth 50% of total
    state['speech_score'] = round((speech_raw / speech_max) * 30, 1)  # Speech worth 30% of total
    integrity_score = round(state['integrity'] * 0.2, 1)  # Integrity worth 20% of total
    state['total_score'] = round(min(state['mcq_score'] + state['speech_score'] + integrity_score, 100), 1)
    state['mcq_raw'] = mcq_raw
    state['speech_raw'] = speech_raw
    state['total_mcq_questions'] = total_mcq_questions
    state['speech_questions_asked'] = speech_asked
    
    if not state.get('topic_covered'): state['topic_covered'] = state.get('jd_topics') or []
    session['page'] = 'result'
    return render_template('index.html', state=state)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    sid = session.get('sid')
    if not sid: session['sid'] = sid = uuid.uuid4().hex
    if sid not in proctor_processors: proctor_processors[sid] = ProctoringProcessor()
    _, ided, gaway = proctor_processors[sid].process_frame(request.get_json()['image'])
    prev = session.get('integrity_deducted', 0)
    new_val = int(ided or 0)
    session['integrity_deducted'] = new_val
    session['gaze_away'] = int(gaway or 0)
    if new_val != prev:
        logging.info(f"Session integrity changed from {prev} to {new_val}")
    return jsonify({'integrity_deducted': ided, 'gaze_away': gaway})

@app.route('/get_current_question')
def get_current_question():
    result = load_result_from_session()
    asked = int(result.get('questions_asked', 0))
    req_list = result.get('required_question', []) or []
    ver_list = result.get('verification_question', []) or []
    # Total questions = actual number of questions available, not the arbitrary max_questions
    total_available = len(req_list) + len(ver_list)
    if total_available == 0: return jsonify({'question': None})
    if asked >= total_available: return jsonify({'question': None})
    
    if asked < len(req_list):
        q_obj = req_list[asked]
        options = q_obj.get('options', [])
        return jsonify({'question': q_obj.get('question'), 'options': options, 'current_question': asked + 1, 'total_questions': total_available})
    
    v_index = asked - len(req_list)
    if v_index < len(ver_list):
        surface = ver_list[v_index].get('surface_level', {})
        return jsonify({'question': surface.get('question'), 'options': surface.get('options', []), 'current_question': asked + 1, 'total_questions': total_available})
    return jsonify({'question': None})

@app.route('/submit_answer', methods=['POST'])
def submit_answer():
    answer = request.get_json().get('answer')
    result = load_result_from_session()
    asked = int(result.get('questions_asked', 0))
    req_list = result.get('required_question', []) or []
    ver_list = result.get('verification_question', []) or []
    correct = False
    
    if asked < len(req_list):
        opts = req_list[asked].get('options', [])
        ans = req_list[asked].get('correct_answer')
        if ans in opts: correct = (answer == opts.index(ans))
    elif asked - len(req_list) < len(ver_list):
        surface = ver_list[asked - len(req_list)].get('surface_level', {})
        opts = surface.get('options', [])
        ans_text = surface.get('correct_answer')
        
        if ans_text and ans_text in opts:
            correct = (answer == opts.index(ans_text))
            
    if correct: result['score'] = result.get('score', 0) + 10
    result['questions_asked'] = asked + 1
    if session.get('result_file'): save_result_to_file(result, filename=session.get('result_file'))
    else: session['result'] = result
    return jsonify({'correct': correct})

@app.route('/get_stats')
def get_stats():
    result = load_result_from_session()
    return jsonify({
        'mcq_score': result.get('score', 0),
        'speech_score': session.get('speech_score', 0),
        'integrity': compute_integrity(100.0, session.get('integrity_deducted', 0)),
        'violations': session.get('integrity_deducted', 0),
        'gaze_away': session.get('gaze_away', 0)
    })

if __name__ == '__main__':
    app.run(debug=True)