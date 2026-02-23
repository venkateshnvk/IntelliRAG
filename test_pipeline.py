from ingestion.pipeline import IngestionPipeline

pipeline = IngestionPipeline()

sample_text = """
Patient ID: PAT1000
Diagnosis: Atrial Fibrillation
Allergies: NSAIDs
"""

pipeline.process_document(
    text=sample_text,
    patient_id="PAT1000",
    report_type="medical",
    source_path="manual_test"
)