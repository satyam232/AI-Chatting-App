from flask import Flask, render_template, request, redirect, url_for
from hashlib import sha256
import json
import time

app = Flask(__name__)

class Block:
    def __init__(self, index, doctor_name, patient_name, medical_report, timestamp, previous_hash):
        self.index = index
        self.doctor_name = doctor_name
        self.patient_name = patient_name
        self.medical_report = medical_report
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        block_string = json.dumps({"index": self.index, "doctor_name": self.doctor_name, "patient_name": self.patient_name,
                                   "medical_report": self.medical_report, "timestamp": self.timestamp, "previous_hash": self.previous_hash}, sort_keys=True).encode()
        return sha256(block_string).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "Genesis Doctor", "Genesis Patient", "No medical reports yet.", time.time(), "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, new_block):
        new_block.previous_hash = self.get_latest_block().hash
        new_block.hash = new_block.calculate_hash()
        self.chain.append(new_block)

blockchain = Blockchain()

@app.route('/', methods=['GET', 'POST'])
def post_medical_report():
    if request.method == 'POST':
        doctor_name = request.form['doctor_name']
        patient_name = request.form['patient_name']
        medical_report = request.form['medical_report']
        new_block = Block(len(blockchain.chain), doctor_name, patient_name, medical_report, time.time(), blockchain.get_latest_block().hash)
        blockchain.add_block(new_block)
        return redirect(url_for('post_medical_report'))
    return render_template('post_medical_report.html')

# Route for user to access medical reports
@app.route('/medr')
def access_medical_reports():
    return render_template('access_medical_reports.html', blockchain=blockchain)

if __name__ == '__main__':
    app.run(port = 5150, debug=True)

