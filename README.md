
# DiscoPDF

## Overview
DiscoPDF is a powerful tool designed for embedding PDF documents into a vectorized database, enabling advanced data interpretation through AI-driven questioning capabilities. This tool primarily uses the Mistral 7B model in GGUF format, though it supports other models as well. It is ideal for researchers and analysts who require deep insights from result data in PDF formats.

## Features
- **PDF Vectorization**: Embed PDF files into a searchable, vectorized database.
- **Flexible Model Support**: Uses Mistral 7B by default but allows the integration of alternative models.
- **Interactive Querying**: Users can query the embedded PDFs to extract and interpret data.
- **Integration Friendly**: Designed to be a part of a broader AI-based workflow for result data interpretation.
- **Streamlit Application**: User-friendly web interface powered by Streamlit.

## Prerequisites
- Python 3.x
- Conda or Miniconda (optional)

## Installation Instructions

### Setting up the Environment
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/nikolatom/DiscoPDF.git
   cd DiscoPDF
   ```

2. **Create a Conda Environment**:
   ```bash
   conda create --name discopdf python=3.8
   conda activate discopdf
   ```

3. **Install Required Python Packages**:
   ```bash
   pip install -r requirements.txt
   ```

### Model Setup
Download the necessary models from Hugging Face as described in the `./models/download_notes.rtf` file included in the repository.

### Running the Application
1. **Launch the Streamlit App**:
   ```bash
   streamlit run disco_pdf.py
   ```

## Usage
Once the application is running, navigate to the local web URL provided by Streamlit (typically `http://localhost:8501`). You can upload PDF files to be embedded into the database and use the interactive interface to query the database.

## Contributing
Contributions to DiscoPDF are welcome! Please refer to the `CONTRIBUTING.md` for guidelines on how to make contributions.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Support
For support, please open an issue in the GitHub repository.

## Acknowledgements
- Models provided by Hugging Face.
- Streamlit for the web framework.
