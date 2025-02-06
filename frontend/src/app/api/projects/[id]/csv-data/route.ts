import { NextResponse } from 'next/server';
import { getCsvData, insertCsvData, updateLabel } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const url = new URL(request.url);
    const fileId = url.searchParams.get('fileId');
    const page = parseInt(url.searchParams.get('page') || '1', 10);
    const pageSize = parseInt(url.searchParams.get('pageSize') || '20', 10);

    if (!fileId) {
      return NextResponse.json({ error: 'File ID is required' }, { status: 400 });
    }

    const csvFileId = parseInt(fileId, 10);
    const result = getCsvData(csvFileId, page, pageSize);
    return NextResponse.json(result);
  } catch (error) {
    console.error('Error fetching CSV data:', error);
    return NextResponse.json({ error: 'Failed to fetch CSV data' }, { status: 500 });
  }
}

export async function POST(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const { csvFileId, data } = await request.json();
    if (!csvFileId || !Array.isArray(data)) {
      return NextResponse.json({ error: 'Invalid data format' }, { status: 400 });
    }

    insertCsvData(csvFileId, data);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error inserting CSV data:', error);
    return NextResponse.json({ error: 'Failed to insert CSV data' }, { status: 500 });
  }
}

export async function PUT(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    const { rowId, labelType, value } = await request.json();
    if (!rowId || !labelType || value === undefined) {
      return NextResponse.json({ error: 'Missing required fields' }, { status: 400 });
    }

    const validLabelTypes = ['human_label', 'llm_label', 'model_label', 'final_label'] as const;
    if (!validLabelTypes.includes(labelType as any)) {
      return NextResponse.json({ error: 'Invalid label type' }, { status: 400 });
    }

    updateLabel(rowId, labelType as typeof validLabelTypes[number], value);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error updating label:', error);
    return NextResponse.json({ error: 'Failed to update label' }, { status: 500 });
  }
}
