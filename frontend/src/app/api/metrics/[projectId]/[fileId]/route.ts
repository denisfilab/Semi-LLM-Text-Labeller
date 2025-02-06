import { NextResponse } from "next/server";

export async function GET(
  request: Request,
  { params }: { params: { projectId: string; fileId: string } }
) {
  try {
    const { projectId, fileId } = await params;

    const response = await fetch(
      `http://localhost:8000/metrics/${projectId}/${fileId}`,
      {
        method: "GET",
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      throw new Error("Failed to fetch metrics");
    }

    const metrics = await response.json();
    return NextResponse.json(metrics);
  } catch (error) {
    console.error("Error fetching metrics:", error);
    return NextResponse.json(
      { error: "Failed to fetch metrics" },
      { status: 500 }
    );
  }
}