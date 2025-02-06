import { NextRequest, NextResponse } from "next/server";
import { getClassificationRules, setClassificationRules } from "@/lib/db";

interface ClassificationRules {
  rules: string;
}

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string; fileId: string } }
) {
  try {
    const fileId = parseInt(params.fileId);
    const rules = getClassificationRules(fileId) as ClassificationRules | undefined;
    return NextResponse.json({ rules: rules?.rules || "" });
  } catch (error) {
    console.error('Error getting classification rules:', error);
    return NextResponse.json(
      { error: 'Failed to get classification rules' },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { id: string; fileId: string } }
) {
  try {
    const fileId = await parseInt(params.fileId);
    const { rules } = await request.json();
    
    if (typeof rules !== 'string') {
      return NextResponse.json(
        { error: 'Rules must be a string' },
        { status: 400 }
      );
    }

    setClassificationRules(fileId, rules);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error setting classification rules:', error);
    return NextResponse.json(
      { error: 'Failed to set classification rules' },
      { status: 500 }
    );
  }
}
