"use client";

import { useEffect, useState } from 'react';
import { Card } from "@/components/ui/card";
import { ProjectSelect } from "@/components/ProjectSelect";
import { CsvFileSelect } from "@/components/CsvFileSelect";
import { useToast } from "@/hooks/use-toast";
import {
  PieChart,
  Pie,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell as RechartsCell,
} from 'recharts';
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface Project {
  id: string;
  name: string;
  createdAt: Date;
}

interface CsvFile {
  id: number;
  filename: string;
  created_at: string;
}

interface Metrics {
  accuracy: number;
  auc_score: number;
  confusion_matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
  sample_counts: {
    train: number;
    test: number;
    total: number;
  };
}

interface InferencePrediction {
  prediction: string;
  confidence: number;
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

/**
 * Improved Confusion Matrix:
 * Displays a table with headers for "Predicted" and "Actual" values.
 */
const ConfusionMatrixTable = ({
  matrix,
}: {
  matrix: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
  };
}) => {
  const total =
    matrix.true_negative +
    matrix.false_positive +
    matrix.false_negative +
    matrix.true_positive;

  // Return a background color based on the cell value proportion.
  const getCellBg = (value: number) => {
    const intensity = Math.min((value / total) * 2, 1);
    return `rgba(59, 130, 246, ${intensity})`;
  };

  // Format a cell: value in bold and percentage in a small text below.
  const formatCell = (value: number) => (
    <div className="flex flex-col items-center justify-center p-2">
      <span className="font-bold text-lg">{value}</span>
      <span className="text-xs text-gray-600">({((value / total) * 100).toFixed(1)}%)</span>
    </div>
  );

  return (
    <div className="overflow-auto">
      <table className="min-w-full border-collapse">
        <thead>
          <tr>
            <th className="p-2 border text-center"></th>
            <th className="p-2 border text-center bg-gray-100">Predicted Positive</th>
            <th className="p-2 border text-center bg-gray-100">Predicted Negative</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th className="p-2 border text-left bg-gray-100">Actual Positive</th>
            <td
              className="border text-center"
              style={{ backgroundColor: getCellBg(matrix.true_positive) }}
            >
              {formatCell(matrix.true_positive)}
            </td>
            <td
              className="border text-center"
              style={{ backgroundColor: getCellBg(matrix.false_negative) }}
            >
              {formatCell(matrix.false_negative)}
            </td>
          </tr>
          <tr>
            <th className="p-2 border text-left bg-gray-100">Actual Negative</th>
            <td
              className="border text-center"
              style={{ backgroundColor: getCellBg(matrix.false_positive) }}
            >
              {formatCell(matrix.false_positive)}
            </td>
            <td
              className="border text-center"
              style={{ backgroundColor: getCellBg(matrix.true_negative) }}
            >
              {formatCell(matrix.true_negative)}
            </td>
          </tr>
        </tbody>
      </table>
      <p className="mt-2 text-center text-sm text-gray-500">
        <strong>Note:</strong> Rows represent the <em>actual</em> class and columns represent the{" "}
        <em>predicted</em> class.
      </p>
    </div>
  );
};

export default function MetricsPage() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([]);
  const [selectedCsvFile, setSelectedCsvFile] = useState<CsvFile | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [inferenceText, setInferenceText] = useState('');
  const [prediction, setPrediction] = useState<InferencePrediction | null>(null);
  const { toast } = useToast();
  const [isLoading, setIsLoading] = useState(false);

  // Load projects
  useEffect(() => {
    fetch('/api/projects')
      .then((res) => res.json())
      .then((data) => {
        if (Array.isArray(data)) {
          setProjects(data);
        }
      })
      .catch((error) => {
        console.error('Error loading projects:', error);
        toast({
          title: 'Error',
          description: 'Failed to load projects',
          variant: 'destructive',
        });
      });
  }, [toast]);

  // Load CSV files when project is selected
  useEffect(() => {
    if (selectedProject) {
      fetch(`/api/projects/${selectedProject.id}/csv-files`)
        .then((res) => res.json())
        .then((data) => {
          if (Array.isArray(data)) {
            setCsvFiles(data);
          }
        })
        .catch((error) => {
          console.error('Error loading CSV files:', error);
          toast({
            title: 'Error',
            description: 'Failed to load CSV files',
            variant: 'destructive',
          });
        });
    } else {
      setCsvFiles([]);
      setSelectedCsvFile(null);
    }
  }, [selectedProject, toast]);

  // Load metrics when file is selected
  useEffect(() => {
    if (selectedProject && selectedCsvFile) {
      setIsLoading(true);
      fetch(`/api/metrics/${selectedProject.id}/${selectedCsvFile.id}`)
        .then((res) => {
          if (!res.ok) throw new Error('Failed to fetch metrics');
          return res.json();
        })
        .then((data) => {
          setMetrics(data);
        })
        .catch((error) => {
          console.error('Error loading metrics:', error);
          toast({
            title: 'Error',
            description:
              "Failed to load metrics. Make sure you've trained a model for this dataset.",
            variant: 'destructive',
          });
          setMetrics(null);
        })
        .finally(() => {
          setIsLoading(false);
        });
    }
  }, [selectedProject, selectedCsvFile, toast]);

  const handleInference = async () => {
    if (!selectedProject || !selectedCsvFile || !inferenceText) return;

    try {
      const response = await fetch('/api/inference', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: selectedProject.id,
          fileId: selectedCsvFile.id,
          text: inferenceText,
        }),
      });

      if (!response.ok) throw new Error('Inference failed');

      const result = await response.json();
      setPrediction(result);
    } catch (error) {
      toast({
        title: 'Error',
        description: 'Failed to get prediction',
        variant: 'destructive',
      });
    }
  };

  return (
    <div className="container mx-auto py-6 px-4 space-y-8">
      <div className="flex flex-col sm:flex-row items-center gap-4">
        <ProjectSelect
          projects={projects}
          selectedProject={selectedProject}
          onSelect={setSelectedProject}
        />
        <CsvFileSelect
          files={csvFiles}
          selectedFile={selectedCsvFile}
          onSelect={setSelectedCsvFile}
        />
      </div>

      {isLoading ? (
        <div className="flex items-center justify-center h-[400px]">
          <div className="flex flex-col items-center gap-4">
            <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
            <p className="text-muted-foreground">Loading metrics...</p>
          </div>
        </div>
      ) : metrics ? (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Performance Cards */}
          <Card className="p-6">
            <h3 className="text-xl font-semibold mb-4">Model Performance</h3>
            <div className="grid grid-cols-2 gap-4">
              <div className="bg-secondary p-4 rounded-lg text-center">
                <p className="text-sm text-gray-600">Accuracy</p>
                <p className="text-2xl font-bold">
                  {(metrics.accuracy * 100).toFixed(1)}%
                </p>
              </div>
              <div className="bg-secondary p-4 rounded-lg text-center">
                <p className="text-sm text-gray-600">AUC Score</p>
                <p className="text-2xl font-bold">
                  {(metrics.auc_score * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </Card>

          {/* Sample Distribution */}
          <Card className="p-6">
            <h3 className="text-xl font-semibold mb-4">Sample Distribution</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={[
                      { name: 'Training', value: metrics.sample_counts.train },
                      { name: 'Testing', value: metrics.sample_counts.test },
                    ]}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                    outerRadius={80}
                    dataKey="value"
                  >
                    {[0, 1].map((entry, index) => (
                      <RechartsCell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </Card>

          {/* Confusion Matrix */}
          <Card className="p-6 col-span-1 lg:col-span-2">
            <h3 className="text-xl font-semibold mb-4">Confusion Matrix</h3>
            <div className="flex items-center justify-center">
              <ConfusionMatrixTable matrix={metrics.confusion_matrix} />
            </div>
          </Card>

          {/* Inference Section */}
          <Card className="p-6 col-span-1 lg:col-span-2">
            <h3 className="text-xl font-semibold mb-4">Try Inference</h3>
            <div className="space-y-4">
              <Input
                placeholder="Enter text to classify..."
                value={inferenceText}
                onChange={(e) => setInferenceText(e.target.value)}
              />
              <Button
                className="w-full"
                onClick={handleInference}
                disabled={!selectedProject || !selectedCsvFile || !inferenceText}
              >
                Predict
              </Button>
              {prediction && (
                <div className="bg-secondary p-4 rounded-lg text-center">
                  <p className="text-sm text-gray-600">Prediction</p>
                  <p className="text-xl font-bold">{prediction.prediction}</p>
                  <p className="text-sm text-gray-600 mt-2">Confidence</p>
                  <p className="text-xl font-bold">
                    {(prediction.confidence * 100).toFixed(1)}%
                  </p>
                </div>
              )}
            </div>
          </Card>
        </div>
      ) : selectedProject && selectedCsvFile ? (
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-center">
            <p className="text-gray-500">
              No metrics available. Train a model for this dataset first.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center h-[400px]">
          <div className="text-center">
            <p className="text-gray-500">
              Select a project and file to view metrics.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
