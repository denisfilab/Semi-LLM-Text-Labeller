import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { projectId, fileId, text } = body;

    if (!projectId || !fileId || !text) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }

    // Forward request to FastAPI backend
    const response = await fetch(`http://localhost:8000/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        project_id: projectId,
        file_id: fileId,
        text: text,
      }),
    });

    if (!response.ok) {
      throw new Error("Prediction failed");
    }

    const result = await response.json();
    return NextResponse.json(result);
  } catch (error) {
    console.error("Error in inference:", error);
    return NextResponse.json(
      { error: "Failed to get prediction" },
      { status: 500 }
    );
  }
} 