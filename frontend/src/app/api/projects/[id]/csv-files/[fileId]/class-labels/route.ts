import { NextRequest, NextResponse } from "next/server";
import { addClassLabel, getClassLabels } from "@/lib/db";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string; fileId: string } }
) {
  try {
    // Await the params before accessing them
    const { fileId } = await params;
    const numericFileId = parseInt(fileId);
    const labels = getClassLabels(numericFileId);
    return NextResponse.json(labels);
  } catch (error) {
    console.error('Error getting class labels:', error);
    return NextResponse.json(
      { error: 'Failed to get class labels' },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string; fileId: string } }
) {
  try {
    // Await the params before accessing them
    const { fileId } = await params;
    const numericFileId = parseInt(fileId);
    const { label } = await request.json();
    
    if (!label) {
      return NextResponse.json(
        { error: 'Label is required' },
        { status: 400 }
      );
    }

    addClassLabel(numericFileId, label);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error adding class label:', error);
    return NextResponse.json(
      { error: 'Failed to add class label' },
      { status: 500 }
    );
  }
}