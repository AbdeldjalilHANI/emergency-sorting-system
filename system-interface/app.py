from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import joblib
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
from sqlalchemy import case
import io
from flask import send_file

app = Flask(__name__, template_folder='templates')

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///patients.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this for production
db = SQLAlchemy(app)

# Updated Patient model
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    symptoms = db.Column(db.String(200), nullable=False)
    severity = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    treated = db.Column(db.Boolean, default=False, nullable=False)
    treated_at = db.Column(db.DateTime)
    
    # Vital signs
    temperature = db.Column(db.Float)
    heart_rate = db.Column(db.Integer)
    blood_pressure = db.Column(db.String(20))
    respiratory_rate = db.Column(db.Integer)
    spo2 = db.Column(db.Integer)
    glasgow = db.Column(db.Integer)
    consciousness = db.Column(db.String(50))
    
    # Medical assessment
    chest_pain_type = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    exercise_angina = db.Column(db.Integer)
    plasma_glucose = db.Column(db.Integer)
    skin_thickness = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    
    # Medical history
    risk_factors = db.Column(db.String(200))
    hypertension = db.Column(db.Integer)
    heart_disease = db.Column(db.Integer)
    massive_bleeding = db.Column(db.Integer)
    respiratory_distress = db.Column(db.Integer)
    residence_type = db.Column(db.String(50))
    smoking_status = db.Column(db.String(50))

class ArchivedPatient(db.Model):
    __tablename__ = 'archived_patients'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    age = db.Column(db.Integer, nullable=False)
    gender = db.Column(db.Integer, nullable=False)
    symptoms = db.Column(db.String(200), nullable=False)
    severity = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    treated = db.Column(db.Boolean, default=False, nullable=False)
    treated_at = db.Column(db.DateTime)
    temperature = db.Column(db.Float)
    heart_rate = db.Column(db.Integer)
    blood_pressure = db.Column(db.String(20))
    respiratory_rate = db.Column(db.Integer)
    spo2 = db.Column(db.Integer)
    glasgow = db.Column(db.Integer)
    consciousness = db.Column(db.String(50))
    chest_pain_type = db.Column(db.Integer)
    cholesterol = db.Column(db.Integer)
    exercise_angina = db.Column(db.Integer)
    plasma_glucose = db.Column(db.Integer)
    skin_thickness = db.Column(db.Integer)
    bmi = db.Column(db.Float)
    risk_factors = db.Column(db.String(200))
    hypertension = db.Column(db.Integer)
    heart_disease = db.Column(db.Integer)
    massive_bleeding = db.Column(db.Integer)
    respiratory_distress = db.Column(db.Integer)
    residence_type = db.Column(db.String(50))
    smoking_status = db.Column(db.String(50))
    archived_at = db.Column(db.DateTime, default=datetime.utcnow)

# Create database tables
with app.app_context():
    db.create_all()

# Load models and encodings
classifier = joblib.load("saved_models/classifier_voting_model.pkl")
regressors = {
    'Resuscitation': joblib.load("saved_models/regressor_level_Resuscitation (L1).pkl"),
    'Emergent': joblib.load("saved_models/regressor_level_Emergent (L2).pkl"),
    'Urgent': joblib.load("saved_models/regressor_level_Urgent (L3).pkl"),
    'Less Urgent': joblib.load("saved_models/regressor_level_Less Urgent (L4).pkl"),
    'Non-Urgent': joblib.load("saved_models/regressor_level_Non-Urgent (L5).pkl")
}

encodings = {
    'Residence_type': {'Rural': 0, 'Urban': 1},
    'smoking_status': {
        'Unknown': 0, 'formerly smoked': 1, 'never smoked': 2, 'smokes': 3
    },
    'Symptom': {
        'Abdominal pain': 0, 'Abdominal pain, Chest pain': 1, 'Abdominal pain, Difficulty breathing': 2,
        'Abdominal pain, Fever': 3, 'Abdominal pain, Headache': 4, 'Abdominal pain, Weakness': 5,
        'Chest pain': 6, 'Chest pain, Abdominal pain': 7, 'Chest pain, Difficulty breathing': 8,
        'Chest pain, Fever': 9, 'Chest pain, Headache': 10, 'Chest pain, Weakness': 11,
        'Difficulty breathing': 12, 'Difficulty breathing, Abdominal pain': 13, 'Difficulty breathing, Chest pain': 14,
        'Difficulty breathing, Fever': 15, 'Difficulty breathing, Headache': 16, 'Difficulty breathing, Weakness': 17,
        'Fever': 18, 'Fever, Abdominal pain': 19, 'Fever, Chest pain': 20, 'Fever, Difficulty breathing': 21,
        'Fever, Headache': 22, 'Fever, Weakness': 23, 'Headache': 24, 'Headache, Abdominal pain': 25,
        'Headache, Chest pain': 26, 'Headache, Difficulty breathing': 27, 'Headache, Fever': 28,
        'Headache, Weakness': 29, 'Weakness': 30, 'Weakness, Abdominal pain': 31,
        'Weakness, Chest pain': 32, 'Weakness, Difficulty breathing': 33,
        'Weakness, Fever': 34, 'Weakness, Headache': 35
    },
    'Consciousness': {'Awake': 0, 'Responds to Pain': 1, 'Unconscious': 2},
    'Risk Factors': {
        'Cancer': 0, 'Cancer, Cardiovascular disease': 1, 'Cancer, Diabetes': 2,
        'Cancer, Hypertension': 3, 'Cancer, None': 4, 'Cancer, kidney failure': 5,
        'Cardiovascular disease': 6, 'Cardiovascular disease, Cancer': 7, 'Cardiovascular disease, Diabetes': 8,
        'Cardiovascular disease, Hypertension': 9, 'Cardiovascular disease, None': 10,
        'Cardiovascular disease, kidney failure': 11, 'Diabetes': 12, 'Diabetes, Cancer': 13,
        'Diabetes, Cardiovascular disease': 14, 'Diabetes, Hypertension': 15, 'Diabetes, None': 16,
        'Diabetes, kidney failure': 17, 'Hypertension': 18, 'Hypertension, Cancer': 19,
        'Hypertension, Cardiovascular disease': 20, 'Hypertension, Diabetes': 21, 'Hypertension, None': 22,
        'Hypertension, kidney failure': 23, 'kidney failure': 24, 'kidney failure, Cancer': 25,
        'kidney failure, Cardiovascular disease': 26, 'kidney failure, Diabetes': 27,
        'kidney failure, Hypertension': 28, 'kidney failure, None': 29, 'none risk factor': 30
    },
    'Massive Bleeding': {False: 0, True: 1},
    'Respiratory Distress': {False: 0, True: 1}
}

def preprocess_and_predict(patient_dict):
    patient = patient_dict.copy()
    
    # Encode categorical fields
    for col, mapping in encodings.items():
        if col in patient:
            patient[col] = mapping[patient[col]]

    # Split blood pressure
    if 'Blood Pressure (mmHg)' in patient:
        bp = patient.pop('Blood Pressure (mmHg)').split('/')
        patient['blood_pressure'] = int(bp[0])
        patient['heart_pressure'] = int(bp[1])

    # Ensure all feature columns are in correct order
    feature_order = [
        'age', 'gender', 'chest pain type', 'cholesterol', 'exercise angina',
        'plasma glucose', 'skin_thickness', 'bmi', 'hypertension', 'heart_disease',
        'Residence_type', 'smoking_status', 'Symptom', 'Temperature (¬∞C)',
        'Heart Rate (bpm)', 'Respiratory Rate (breaths/min)', 'SpO2 (%)', 'Glasgow Score',
        'Consciousness', 'Massive Bleeding', 'Respiratory Distress',
        'Risk Factors', 'blood_pressure', 'heart_pressure'
    ]
    
    patient_df = pd.DataFrame([patient])[feature_order]
    cls = classifier.predict(patient_df)[0]
    score = regressors[cls].predict(patient_df)[0]

    return cls, score

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        form_data = request.form.to_dict()
        
        # Prepare patient data for model
        patient_data = {
            'age': float(form_data.get('age')),
            'gender': int(form_data.get('gender')),
            'chest pain type': float(form_data.get('chest_pain_type')),
            'cholesterol': float(form_data.get('cholesterol')),
            'exercise angina': float(form_data.get('exercise_angina')),
            'plasma glucose': float(form_data.get('plasma_glucose')),
            'skin_thickness': float(form_data.get('skin_thickness')),
            'bmi': float(form_data.get('bmi')),
            'hypertension': float(form_data.get('hypertension')),
            'heart_disease': float(form_data.get('heart_disease')),
            'Residence_type': form_data.get('Residence'),
            'smoking_status': form_data.get('smoking'),
            'Symptom': form_data.get('Symptom'),
            'Temperature (¬∞C)': float(form_data.get('Temperature')),
            'Heart Rate (bpm)': float(form_data.get('heart_rate')),
            'Respiratory Rate (breaths/min)': float(form_data.get('Respiratory_Rate')),
            'SpO2 (%)': float(form_data.get('spo2')),
            'Glasgow Score': float(form_data.get('glasgow')),
            'Consciousness': form_data.get('Consciousness'),
            'Massive Bleeding': int(form_data.get('Massive_Bleeding')),
            'Respiratory Distress': int(form_data.get('Resp_Distress')),
            'Risk Factors': form_data.get('Risk'),
            'Blood Pressure (mmHg)': form_data.get('bp')
        }
        
        # Make prediction
        severity, score = preprocess_and_predict(patient_data)
        
        # Save to database with all details
        new_patient = Patient(
            name=form_data.get('name'),
            age=form_data.get('age'),
            gender=form_data.get('gender'),
            symptoms=form_data.get('Symptom'),
            severity=severity,
            score=float(score),
            temperature=form_data.get('Temperature'),
            heart_rate=form_data.get('heart_rate'),
            blood_pressure=form_data.get('bp'),
            respiratory_rate=form_data.get('Respiratory_Rate'),
            spo2=form_data.get('spo2'),
            glasgow=form_data.get('glasgow'),
            consciousness=form_data.get('Consciousness'),
            chest_pain_type=form_data.get('chest_pain_type'),
            cholesterol=form_data.get('cholesterol'),
            exercise_angina=form_data.get('exercise_angina'),
            plasma_glucose=form_data.get('plasma_glucose'),
            skin_thickness=form_data.get('skin_thickness'),
            bmi=form_data.get('bmi'),
            risk_factors=form_data.get('Risk'),
            hypertension=form_data.get('hypertension'),
            heart_disease=form_data.get('heart_disease'),
            massive_bleeding=form_data.get('Massive_Bleeding'),
            respiratory_distress=form_data.get('Resp_Distress'),
            residence_type=form_data.get('Residence'),
            smoking_status=form_data.get('smoking')
        )
        
        db.session.add(new_patient)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'name': form_data.get('name'),
            'severity': severity,
            'score': score
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/records')
def records():
    # Define custom severity order
    severity_order = case(
        {
            'Resuscitation': 1,
            'Emergent': 2,
            'Urgent': 3,
            'Less Urgent': 4,
            'Non-Urgent': 5
        },
        value=Patient.severity
    )

    try:
        if hasattr(Patient, 'treated'):
            patients = Patient.query.filter_by(treated=False)\
                .order_by(severity_order, Patient.score.desc())\
                .all()
        else:
            patients = Patient.query.order_by(severity_order, Patient.score.desc()).all()
    except Exception as e:
        print(f"Error querying patients: {e}")
        patients = Patient.query.order_by(Patient.score.desc()).all()
        
    return render_template('records.html', patients=patients)

@app.route('/get_patients', methods=['GET'])
def get_patients():
    patients = Patient.query.order_by(Patient.score.desc()).all()
    patients_data = [{
        'id': p.id,
        'name': p.name,
        'age': p.age,
        'symptoms': p.symptoms,
        'severity': p.severity,
        'score': p.score,
        'timestamp': p.timestamp.strftime('%Y-%m-%d %H:%M')
    } for p in patients]
    return jsonify(patients_data)





@app.route('/patient/<int:patient_id>')
def patient_details(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    return render_template('patient_details.html', patient=patient)

@app.route('/mark_treated/<int:patient_id>', methods=['POST'])
def mark_treated(patient_id):
    patient = Patient.query.get_or_404(patient_id)
    patient.treated = True
    patient.treated_at = datetime.utcnow()
    db.session.commit()
    
    # Check if the request wants JSON response
    if request.headers.get('Content-Type') == 'application/json':
        return jsonify({'success': True})
    else:
        return redirect(url_for('history'))

@app.route('/history')
def history():
    try:
        treated_patients = Patient.query.filter_by(treated=True).order_by(Patient.treated_at.desc()).all()
    except:
        # Fallback if treated column doesn't exist yet
        treated_patients = []
    return render_template('history.html', patients=treated_patients)


# Add new route
@app.route('/reports')
def reports():
    # Get filter parameters
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    selected_severity = request.args.get('severity', '')

    # Base query
    query = Patient.query

    # Apply date filter
    if start_date and end_date:
        try:
            start = datetime.strptime(start_date, '%Y-%m-%d')
            end = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=1)
            query = query.filter(Patient.timestamp.between(start, end))
        except ValueError:
            pass

    # Apply severity filter
    if selected_severity:
        query = query.filter_by(severity=selected_severity)

    # Get filtered patients
    filtered_patients = query.all()

    # Calculate statistics
    total_patients = len(filtered_patients)
    active_cases = sum(1 for p in filtered_patients if not p.treated)
    treated_cases = total_patients - active_cases
    
    # Treatment time calculation
    treatment_times = []
    for p in filtered_patients:
        if p.treated and p.treated_at:
            delta = p.treated_at - p.timestamp
            treatment_times.append(delta.total_seconds())
    
    avg_seconds = sum(treatment_times)/len(treatment_times) if treatment_times else 0
    avg_td = timedelta(seconds=avg_seconds)
    hours, remainder = divmod(avg_td.seconds, 3600)
    minutes = remainder // 60
    avg_treatment_time = f"{hours}h {minutes}m" if avg_td else "N/A"

    # Treated/Pending percentages
    treated_percentage = (treated_cases / total_patients * 100) if total_patients else 0
    pending_percentage = 100 - treated_percentage

    # Severity distribution
    severity_distribution = {
        'Resuscitation': 0,
        'Emergent': 0,
        'Urgent': 0,
        'Less Urgent': 0,
        'Non-Urgent': 0
    }
    for p in filtered_patients:
        severity_distribution[p.severity] += 1

    return render_template(
        'reports.html',
        total_patients=total_patients,
        active_cases=active_cases,
        treated_cases=treated_cases,
        avg_treatment_time=avg_treatment_time,
        treated_percentage=treated_percentage,
        pending_percentage=pending_percentage,
        severity_distribution=severity_distribution,
        start_date=start_date,
        end_date=end_date,
        selected_severity=selected_severity
    )

from sqlalchemy import inspect

@app.route('/export-history')
def export_history():
    try:
        # Check if any patients exist
        patients = Patient.query.filter_by(treated=True).all()
        if not patients:
            return "No patients to export", 404

        # Create DataFrame
        patient_data = []
        for p in patients:
            patient_dict = {c.name: getattr(p, c.name) for c in Patient.__table__.columns}
            patient_dict.pop('id')  # Remove the original ID
            patient_data.append(patient_dict)

        df = pd.DataFrame(patient_data)

        # Create Excel file in memory
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='Patients', index=False)
        
        # Move patients to archive and delete
        for p in patients:
            archived = ArchivedPatient(**{c.name: getattr(p, c.name) for c in ArchivedPatient.__table__.columns 
                                      if c.name != 'id' and c.name != 'archived_at'})
            db.session.add(archived)
            db.session.delete(p)
        
        db.session.commit()

        # Prepare response
        output.seek(0)
        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            download_name='patient_history.xlsx',
            as_attachment=True
        )

    except Exception as e:
        db.session.rollback()
        return f"Error occurred: {str(e)}", 500

@app.route('/archive')
def archive():
    archived_patients = ArchivedPatient.query.order_by(ArchivedPatient.treated_at.desc()).all()
    return render_template('archive.html', patients=archived_patients)

if __name__ == '__main__':
    app.run(debug=True)