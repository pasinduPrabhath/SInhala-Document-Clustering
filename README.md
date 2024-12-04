# Sinhala Named Entity Recognition (NER) and Document Clustering

This repository contains a Flask-based web application for Sinhala Named Entity Recognition (NER) and document clustering. The application uses a hybrid NER process that combines a pre-trained SpaCy model and the Perplexity API (LLaMa 3.1 model) for enhanced entity detection. For document clustering, it employs a Word2Vec model and fuzzy clustering (Fuzzy C-Means) with the elbow method to determine the optimal number of clusters. The clusters are then named using the LLaMa 3.1 model from Perplexity API.


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Configuration](#configuration)

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/sinhala_ner.git
    cd sinhala_ner
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

4. Create a [.env](http://_vscodecontentref_/2) file in the root directory and add your Perplexity API key:
    ```env
    PERPLEXITY_API_KEY=your_api_key_here
    ```

## Usage

1. Start the Flask application:
    ```sh
    python app.py
    ```

2. The application will be available at `http://127.0.0.1:5000`.

## API Endpoints

### `/detect_ner`

- **Method:** `POST`
- **Description:** Detects named entities in the provided sentences.
- **Request Body:**
    ```json
    {
        "sentences": ["sentence1", "sentence2"]
    }
    ```
- **Response:**
    ```json
    {
        "entities": [
            {
                "sentence": "sentence1",
                "detected_entities": [
                    {"text": "entity1", "type": "PERSON"}
                ]
            }
        ]
    }
    ```

### `/detect_and_cluster`

- **Method:** `POST`
- **Description:** Detects named entities and clusters the sentences based on the detected entities.
- **Request Body:**
    ```json
    {
        "sentences": ["sentence1", "sentence2"]
    }
    ```
- **Response:**
    ```json
    {
        "total_clusters": 3,
        "clusters": [
            {
                "cluster_name": "Cluster 1",
                "sentences": [
                    {"sentence": "sentence1", "membership_probability": 0.9}
                ]
            }
        ]
    }
    ```

## Configuration

Configuration settings are stored in the [config.py](http://_vscodecontentref_/3) file and the [.env](http://_vscodecontentref_/4) file.

- **config.py:**
    ```python
    PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY")
    SPACY_MODEL_PATH = "spacy_model"
    ```

- **.env:**
    ```env
    PERPLEXITY_API_KEY=your_api_key_here
    ```
