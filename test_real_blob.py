from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

files = [
    "01_medical_reports/PAT1000_medical_report.pdf",
    "02_scanning_reports/PAT1000_scan.json",
    "03_medication_records/PAT1000_meds.docx",
    "04_patient_billings/PAT1000_billing_statement.pdf",  # ✅ correct
    "05_insurance_policies/PAT1000_insurance_policy.html"
]

for file in files:
    print(f"\nProcessing {file}")
    pipeline.process_blob(file)