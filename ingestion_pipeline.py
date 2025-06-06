import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer, util
from keybert import KeyBERT
from openai import AzureOpenAI
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from pinecone import Pinecone, ServerlessSpec


class MetadataExtractionPipeline:
    def __init__(self):
        load_dotenv(dotenv_path=".env")

        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME")

        if self.index_name not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=self.index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-west-2")
            )

        self.pinecone_index = self.pc.Index(self.index_name)

        self.pdf_input_dir = Path("data")
        self.final_output_path = Path("output/metadata_tagging.json")
        self.final_output_path.parent.mkdir(parents=True, exist_ok=True)

        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.keyword_model = KeyBERT()
        self.ner_pipeline = pipeline(
            "ner",
            model=AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/roberta-large-ner-english"),
            tokenizer=AutoTokenizer.from_pretrained("Jean-Baptiste/roberta-large-ner-english"),
            aggregation_strategy="simple"
        )

        self.intent_examples = self.load_intent_examples("data/intent_examples.json")

    def load_intent_examples(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_text_from_pdf(self, pdf_path):
        reader = PdfReader(str(pdf_path))
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text.strip()

    def clean_text(self, text):
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'[^\w\s\-]', '', text)
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip().lower()

    def extract_keywords(self, text):
        cleaned = self.clean_text(text)
        keywords = [kw for kw, _ in self.keyword_model.extract_keywords(
            cleaned,
            top_n=10,
            stop_words='english',
            keyphrase_ngram_range=(1, 1)
        )]
        return keywords

    def predict_intent(self, text):
        labels = list(self.intent_examples.keys())
        examples = list(self.intent_examples.values())

        doc_emb = self.embedding_model.encode(text, convert_to_tensor=True)
        example_emb = self.embedding_model.encode(examples, convert_to_tensor=True)
        cosine_scores = util.cos_sim(doc_emb, example_emb)[0]

        best_idx = int(cosine_scores.argmax())
        return labels[best_idx], float(cosine_scores[best_idx])

    def extract_entities(self, text):
        return [{"text": ent["word"], "label": ent["entity_group"]} for ent in self.ner_pipeline(text)]

    def summarize(self, text):
        prompt = f"Summarize the following text in 2–3 lines:\n\n{text[:3000]}"
        try:
            response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=256
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error in summarization: {e}")
            return ""

    def embed_text(self, text):
        return self.embedding_model.encode(text).tolist()

    def run_pipeline(self):
        results = []

        for pdf_file in self.pdf_input_dir.glob("*.pdf"):
            print(f"📄 Processing: {pdf_file.name}")
            text = self.extract_text_from_pdf(pdf_file)

            if not text:
                print(f"⚠️ Skipped: {pdf_file.name} (empty or unreadable)")
                continue

            intent, confidence = self.predict_intent(text)

            metadata = {
                "filename": pdf_file.name,
                "keywords": self.extract_keywords(text),
                "named_entities": self.extract_entities(text),
                "summary": self.summarize(text),
                "intent_category": intent,
                "intent_confidence": confidence,
                "embedding": self.embed_text(text)
            }

            results.append(metadata)

            entity_strings = [f"{ent['text']} ({ent['label']})" for ent in metadata["named_entities"]]

            self.pinecone_index.upsert([
                (
                    f"{pdf_file.stem}",
                    metadata["embedding"],
                    {
                        "filename": metadata["filename"],
                        "intent": metadata["intent_category"],
                        "summary": metadata["summary"],
                        "keywords": metadata["keywords"],
                        "named_entities": entity_strings
                    }
                )
            ])

            print(f"✅ Done: {pdf_file.name}")

        with open(self.final_output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n📦 All metadata saved to {self.final_output_path}")


# 🚀 Entry Point
if __name__ == "__main__":
    pipeline = MetadataExtractionPipeline()
    pipeline.run_pipeline()
