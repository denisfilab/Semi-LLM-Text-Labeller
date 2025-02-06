# Semi LLM Text Labeller

A full-stack application for semi-automated text labeling using Large Language Models (LLMs). The system combines machine learning with human oversight to efficiently label text data.

## Project Structure

```
.
├── app/                # Backend FastAPI application
├── frontend/          # Next.js frontend application
├── database/          # Database files and migrations
└── results/           # Output results directory
```

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm or yarn
- PostgreSQL database

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd Semi-LLM-Text-Labeller
```

2. Set up the backend:
```bash
cd app
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env     # Configure your environment variables
```

3. Set up the frontend:
```bash
cd frontend
npm install
cp .env.example .env.local  # Configure your environment variables
```

4. Start the services:

Backend:
```bash
cd app
uvicorn main:app --reload
```

Frontend:
```bash
cd frontend
npm run dev
```

The application will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

## Features

- Semi-automated text labeling using LLMs
- Real-time labeling progress tracking
- Interactive labeling interface
- Data export and analytics
- Model performance metrics
- Batch processing capabilities

## Documentation

- Backend API documentation is available at `/docs` when the server is running
- Detailed setup instructions are available in the respective README files in the `app/` and `frontend/` directories

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details 