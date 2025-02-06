// File: /app/api/test-pipeline/route.ts (Next.js 13+ folder-based routing)
import { NextRequest, NextResponse } from "next/server";

interface PipelineStage {
  name: string;
  status: "pending" | "in_progress" | "completed";
  progress: number; // 0-100
  description: string;
}

// --- Global Variables ---
// These persist ONLY as long as your server stays in memory.
let pipelineStartTime: number | null = null;
let pipelineStatus: "pending" | "in_progress" | "completed" | "failed" = "pending";
let jobId: string | null = null;

const stages: PipelineStage[] = [
  {
    name: "Embedding Generation",
    status: "pending",
    progress: 0,
    description: "Generating embeddings for text samples..."
  },
  {
    name: "Clustering",
    status: "pending",
    progress: 0,
    description: "Clustering samples for diversity..."
  },
  {
    name: "LLM Labeling",
    status: "pending",
    progress: 0,
    description: "Applying LLM-based labeling..."
  },
  {
    name: "Model Training",
    status: "pending",
    progress: 0,
    description: "Training classification model..."
  }
];

// For cost estimate demo
const mockCostEstimate = {
  totalTokens: 150000,
  promptTokens: 100000,
  completionTokens: 50000,
  estimatedCost: 2.50,
  sampleCount: 1000
};

// Duration config
// We have 4 stages * 5 seconds each = 20s total.
const totalStages = stages.length;
const stageDurationMs = 5000; // 5 seconds per stage => 20 seconds total

// Utility to reset pipeline state (for repeated testing)
function resetPipeline(): void {
  pipelineStartTime = null;
  pipelineStatus = "pending";
  jobId = null;
  stages.forEach(stage => {
    stage.status = "pending";
    stage.progress = 0;
  });
}

export async function GET(request: NextRequest) {
  // Just explain how to use endpoints
  return NextResponse.json({
    message: "Mock pipeline with 4 stages. See POST requests for cost or pipeline actions.",
    usage: {
      POST: {
        "type=cost": "Returns a mock cost estimate.",
        "type=init": "Resets and starts a new pipeline. Returns { jobId }. 20s total duration.",
        "type=status": "Returns the progress of the pipeline run.",
        "type=reset": "Resets pipeline state to 'pending'."
      }
    }
  });
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { type } = body;

    // 1. Return a mock cost estimate
    if (type === "cost") {
      return NextResponse.json(mockCostEstimate);
    }

    // 2. Initialize (start) the pipeline
    if (type === "init") {
      // Reset the pipeline state so we can run it fresh
      resetPipeline();

      pipelineStartTime = Date.now();
      pipelineStatus = "in_progress";
      jobId = Math.random().toString(36).substring(2, 10); // random small string

      return NextResponse.json({
        jobId,
        message: "Pipeline started. Will take ~20 seconds total."
      });
    }

    // 3. Check pipeline status
    if (type === "status") {
      if (!pipelineStartTime || !jobId) {
        return NextResponse.json({ 
          error: "No pipeline is currently running. Call type=init first." 
        }, { status: 400 });
      }

      const elapsed = Date.now() - pipelineStartTime;
      const currentStageIndex = Math.floor(elapsed / stageDurationMs);

      // If we've passed the last stage, pipeline is complete
      if (currentStageIndex >= totalStages && pipelineStatus !== "completed") {
        pipelineStatus = "completed";
        stages.forEach(stage => {
          stage.status = "completed";
          stage.progress = 100;
        });
      } else if (pipelineStatus !== "completed") {
        // Update each stage's status and progress
        for (let i = 0; i < totalStages; i++) {
          if (i < currentStageIndex) {
            // Completed stages
            stages[i].status = "completed";
            stages[i].progress = 100;
          } else if (i === currentStageIndex) {
            // Current stage
            stages[i].status = "in_progress";
            // Progress within the current stage, in percentage
            const stageElapsed = elapsed % stageDurationMs;
            const stageProgressPct = Math.floor((stageElapsed / stageDurationMs) * 100);
            stages[i].progress = stageProgressPct;
          } else {
            // Future stages
            stages[i].status = "pending";
            stages[i].progress = 0;
          }
        }
      }

      return NextResponse.json({
        jobId,
        status: pipelineStatus,
        stages
      });
    }

    // 4. Reset pipeline (optional helper)
    if (type === "reset") {
      resetPipeline();
      return NextResponse.json({ message: "Pipeline reset to pending." });
    }

    // If we reach here, invalid type
    return NextResponse.json(
      { error: "Invalid request body 'type'." },
      { status: 400 }
    );
  } catch (error) {
    console.error("Error in test pipeline:", error);
    return NextResponse.json(
      { error: "Test pipeline error" },
      { status: 500 }
    );
  }
}
