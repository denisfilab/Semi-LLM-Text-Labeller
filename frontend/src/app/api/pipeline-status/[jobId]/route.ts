import { NextRequest, NextResponse } from "next/server";

export async function GET(
  request: NextRequest,
  { params }: { params: { jobId: string } }
) {
  params = await params;
  const jobId = await params.jobId;

  try {
    if (!jobId) {
      return NextResponse.json(
        { error: "Job ID is required" },
        { status: 400 }
      );
    }

    const response = await fetch(`http://localhost:8000/pipeline-status/${jobId}`);
    
    if (!response.ok) {
      throw new Error("Failed to get pipeline status");
    }

    const status = await response.json();
    console.log('[API] Raw status from backend:', JSON.stringify(status, null, 2));

    // The backend now returns the exact structure we need, so we can pass it through
    return NextResponse.json(status);
  } catch (error) {
    console.error("Error getting pipeline status:", error);
    return NextResponse.json(
      { error: "Failed to get pipeline status" },
      { status: 500 }
    );
  }
}