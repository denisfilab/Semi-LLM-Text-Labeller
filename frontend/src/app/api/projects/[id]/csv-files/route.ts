import { NextResponse } from 'next/server';
import { createCsvFile, getCsvFiles, getProject } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Await the entire params object first
    const { id } = await params;
    const files = getCsvFiles(id);
    return NextResponse.json(files);
  } catch (error) {
    console.error('Error fetching CSV files:', error);
    return NextResponse.json({ error: 'Failed to fetch CSV files' }, { status: 500 });
  }
}

export async function POST(
  request: Request,
  { params }: { params: { id: string } }
) {
  try {
    // Await the entire params object first
    const { id } = await params;
    
    // Check if project exists first
    const project = getProject(id);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const { filename } = await request.json();
    if (!filename) {
      return NextResponse.json({ error: 'Filename is required' }, { status: 400 });
    }

    const fileId = createCsvFile(id, filename);
    return NextResponse.json({ id: fileId, filename });
  } catch (error) {
    console.error('Error creating CSV file:', error);
    return NextResponse.json({ error: 'Failed to create CSV file' }, { status: 500 });
  }
}