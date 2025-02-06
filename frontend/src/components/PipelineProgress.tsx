"use client";

import { Progress } from "@/components/ui/progress";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { CheckCircle2, Circle, Loader2 } from "lucide-react";

interface Stage {
  name: string;
  status: "pending" | "in_progress" | "completed";
  progress: number;
  description: string;
}

interface PipelineProgressProps {
  stages: Stage[];
  isVisible: boolean;
}

export function PipelineProgress({ stages, isVisible }: PipelineProgressProps) {
  if (!isVisible) return null;

  const getStatusIcon = (status: Stage["status"]) => {
    switch (status) {
      case "completed":
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case "in_progress":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      default:
        return <Circle className="h-4 w-4 text-muted-foreground" />;
    }
  };

  const getStatusBadge = (status: Stage["status"]) => {
    switch (status) {
      case "completed":
        return (
          <Badge variant="default" className="bg-green-500">
            Completed
          </Badge>
        );
      case "in_progress":
        return (
          <Badge variant="default" className="bg-blue-500">
            In Progress
          </Badge>
        );
      default:
        return (
          <Badge variant="outline">
            Pending
          </Badge>
        );
    }
  };

  return (
    <Card className="p-6">
      <h3 className="text-lg font-semibold mb-4">Pipeline Progress</h3>
      <div className="space-y-6">
        {stages.map((stage, index) => (
          <div key={stage.name} className="space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getStatusIcon(stage.status)}
                <span className="font-medium">{stage.name}</span>
              </div>
              {getStatusBadge(stage.status)}
            </div>
            <Progress value={stage.progress} className="h-2" />
            <p className="text-sm text-muted-foreground">{stage.description}</p>
            {index < stages.length - 1 && <div className="border-t my-4" />}
          </div>
        ))}
      </div>
    </Card>
  );
}
