import { NextRequest, NextResponse } from "next/server";
import { getDb } from "@/lib/db";

const FASTAPI_URL = "http://localhost:8000"; // FastAPI backend URL

interface CsvRow {
  text: string;
}

export async function POST(request: NextRequest) {
  try {
    const { projectId, columnName, fileId } = await request.json();

    if (!projectId || !columnName || !fileId) {
      return NextResponse.json(
        { error: "Missing required parameters" },
        { status: 400 }
      );
    }

    // Get CSV data directly from SQLite database
    const db = getDb();
    const rows = db.prepare('SELECT text FROM csv_data WHERE csv_file_id = ?').all(fileId) as CsvRow[];
    
    if (!rows || rows.length === 0) {
      return NextResponse.json(
        { error: "No data found for the specified file" },
        { status: 404 }
      );
    }

  
    const header = columnName;  // e.g., "text"

    // 2. Transform each row’s text -> a quoted+escaped string
    //    so commas and line breaks don’t break the CSV structure
    const formattedRows = rows.map(({ text }) => {
      // Escape internal quotes by doubling them
      const escapedText = text.replace(/"/g, '""');
      // Wrap the entire text in double quotes
      return `"${escapedText}"`;
    });
    
    
    // Create form data with file
    const formData = new FormData();
    formData.append('project_id', projectId);
    formData.append('column_name', columnName);
    formData.append('file_id', fileId);

    // Forward the request to FastAPI backend
    const response = await fetch(`${FASTAPI_URL}/run-pipeline`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || "Pipeline initialization failed" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error in run-pipeline:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}

// Get pipeline status
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const jobId = searchParams.get("jobId");

    if (!jobId) {
      return NextResponse.json(
        { error: "Missing job ID" },
        { status: 400 }
      );
    }

    const response = await fetch(`${FASTAPI_URL}/pipeline-status/${jobId}`);
    
    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { error: error.detail || "Failed to get pipeline status" },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error("Error getting pipeline status:", error);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
