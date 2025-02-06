import { NextResponse } from 'next/server';
import { addClassLabel, getClassLabels, getProject } from '@/lib/db';

export async function GET(
  request: Request,
  { params }: { params: { id: string } }
) {
  const projectId = await Promise.resolve(params.id);
  try {
    const labels = getClassLabels(projectId);
    return NextResponse.json(labels);
  } catch (error) {
    console.error('Error fetching class labels:', error);
    return NextResponse.json({ error: 'Failed to fetch class labels' }, { status: 500 });
  }
}

export async function POST(
  request: Request,
  { params }: { params: { id: string } }
) {
  const projectId = await Promise.resolve(params.id);
  try {
    // Check if project exists first
    const project = getProject(projectId);
    if (!project) {
      return NextResponse.json({ error: 'Project not found' }, { status: 404 });
    }

    const { label } = await request.json();
    if (!label) {
      return NextResponse.json({ error: 'Label is required' }, { status: 400 });
    }
    addClassLabel(projectId, label);
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error('Error adding class label:', error);
    return NextResponse.json({ error: 'Failed to add class label' }, { status: 500 });
  }
}
