"use client";

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Card } from "@/components/ui/card";

interface CostEstimate {
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  estimatedCost: number;
  sampleCount: number;
}

interface CostEstimateModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  estimate: CostEstimate | null;
  isLoading: boolean;
}

export function CostEstimateModal({
  isOpen,
  onClose,
  onConfirm,
  estimate,
  isLoading,
}: CostEstimateModalProps) {
  if (!estimate) return null;

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Processing Cost Estimate</DialogTitle>
          <DialogDescription>
            Review the estimated cost for processing your dataset with our LLM pipeline.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-4 py-4">
          <Card className="p-4">
            <h3 className="font-semibold mb-2">Dataset Overview</h3>
            <p className="text-sm text-muted-foreground mb-4">
              {estimate.sampleCount.toLocaleString()} samples to process
            </p>
            
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm">Prompt Tokens:</span>
                <span className="font-medium">{estimate.promptTokens.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Completion Tokens:</span>
                <span className="font-medium">{estimate.completionTokens.toLocaleString()}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm">Total Tokens:</span>
                <span className="font-medium">{estimate.totalTokens.toLocaleString()}</span>
              </div>
              <div className="pt-2 border-t">
                <div className="flex justify-between items-center">
                  <span className="font-semibold">Estimated Cost:</span>
                  <span className="text-lg font-bold">
                    ${estimate.estimatedCost.toFixed(2)}
                  </span>
                </div>
              </div>
            </div>
          </Card>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <Button onClick={onConfirm} disabled={isLoading}>
            {isLoading ? "Processing..." : "Confirm & Process"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
