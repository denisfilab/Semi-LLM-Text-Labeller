"use client";

import Papa from "papaparse";
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { ProjectSelect } from "@/components/ProjectSelect"
import { CsvFileSelect } from "@/components/CsvFileSelect"
import { useEffect, useState } from "react";
import { useToast } from "@/hooks/use-toast"
import Link from "next/link";
import { DataTable } from "@/components/DataTable";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { CostEstimateModal } from "@/components/CostEstimateModal";
import { PipelineProgress } from "@/components/PipelineProgress";
import { ToastProvider } from "@/components/toast";

interface CsvRow {
  id: number;
  text: string;
  llm_label?: string;
  model_label?: string;
  human_label?: string;
  final_label?: string;
}

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

interface CostEstimate {
  totalTokens: number;
  promptTokens: number;
  completionTokens: number;
  estimatedCost: number;
  sampleCount: number;
}

interface PipelineStage {
  name: string;
  status: "pending" | "in_progress" | "completed";
  progress: number;
  description: string;
}

export default function HomePage() {
  const [classes, setClasses] = useState<string[]>([])
  const [newClass, setNewClass] = useState("")
  const [rules, setRules] = useState("")
  const [rulesLoaded, setRulesLoaded] = useState(false)
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [csvData, setCsvData] = useState<CsvRow[]>([])
  const [hoveredRowIndex, setHoveredRowIndex] = useState<number | null>(null)
  const [projects, setProjects] = useState<Project[]>([])
  const [selectedProject, setSelectedProject] = useState<Project | null>(null)
  const [csvFiles, setCsvFiles] = useState<CsvFile[]>([])
  const [selectedCsvFile, setSelectedCsvFile] = useState<CsvFile | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState(20)
  const [totalPages, setTotalPages] = useState(1)
  const [costEstimate, setCostEstimate] = useState<CostEstimate | null>(null)
  const [showCostModal, setShowCostModal] = useState(false)
  const [pipelineStages, setPipelineStages] = useState<PipelineStage[]>([
    {
      name: "Embedding Generation",
      status: "pending",
      progress: 0,
      description: "Generating embeddings for text samples..."
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
  ])
  const [llmLabelingCompleted, setLlmLabelingCompleted] = useState(false);

  const [showProgress, setShowProgress] = useState(false)
  const { toast } = useToast()

  // Load projects
  useEffect(() => {
    fetch('/api/projects')
      .then(res => res.json())
      .then(data => {
        if (!Array.isArray(data)) {
          console.error('Invalid projects data:', data);
          return;
        }
        setProjects(data);
      })
      .catch(error => {
        console.error('Error loading projects:', error);
        toast({
          title: "Error",
          description: "Failed to load projects",
          variant: "destructive",
        });
      });
  }, []);

  // Load CSV files when project is selected
  useEffect(() => {
    if (selectedProject) {
      fetch(`/api/projects/${selectedProject.id}/csv-files`)
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data)) {
            setCsvFiles(data);
          }
        })
        .catch(error => {
          console.error('Error loading CSV files:', error);
          toast({
            title: "Error",
            description: "Failed to load CSV files",
            variant: "destructive",
          });
        });
    } else {
      setCsvFiles([]);
      setSelectedCsvFile(null);
    }
  }, [selectedProject]);

  // Load CSV data when file is selected
  useEffect(() => {
    if (selectedCsvFile) {
      loadCsvData();
    } else {
      setCsvData([]);
    }
  }, [selectedCsvFile, page, pageSize]);

  const loadCsvData = async () => {
    if (!selectedCsvFile) {
      setCsvData([]);
      setTotalPages(1);
      return;
    }

    try {
      const response = await fetch(`/api/projects/${selectedProject?.id}/csv-data?fileId=${selectedCsvFile.id}&page=${page}&pageSize=${pageSize}`);
      if (!response.ok) {
        throw new Error('Failed to load CSV data');
      }
      const result = await response.json();
      setCsvData(result.data);
      setTotalPages(result.totalPages);
    } catch (error) {
      console.error('Error loading CSV data:', error);
      toast({
        title: "Error",
        description: "Failed to load CSV data",
        variant: "destructive",
      });
    }
  };

  const handleProcessCsv = async () => {
    if (!selectedFile || !selectedProject) return;
    
    setIsProcessing(true);
    
    try {
      const parseResult = await new Promise<Papa.ParseResult<CsvRow>>((resolve, reject) => {
        Papa.parse(selectedFile, {
          header: true,
          skipEmptyLines: true,
          complete: resolve,
          error: reject,
        });
      });

      const headers = parseResult.meta.fields ?? [];
      if (!headers.includes("text")) {
        throw new Error('CSV file missing required "text" column');
      }

      // Format the parsed data
      const parsedData = parseResult.data.map(row => ({
        text: row.text || "",
        llm_label: row.llm_label || "",
        model_label: row.model_label || "",
        human_label: row.human_label || "",
        final_label: row.final_label || "",
      }));

      // Create CSV file record
      const fileResponse = await fetch(`/api/projects/${selectedProject.id}/csv-files`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedFile.name }),
      });

      if (!fileResponse.ok) {
        throw new Error('Failed to create CSV file record');
      }

      const fileData = await fileResponse.json();
      const csvFileId = fileData.id;

      // Insert CSV data into database
      const dataResponse = await fetch(`/api/projects/${selectedProject.id}/csv-data`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          csvFileId: csvFileId,
          data: parsedData
        }),
      });

      if (!dataResponse.ok) {
        throw new Error('Failed to insert CSV data');
      }

      // Refresh CSV files list and select the new file
      const filesResponse = await fetch(`/api/projects/${selectedProject.id}/csv-files`);
      if (filesResponse.ok) {
        const files = await filesResponse.json();
        setCsvFiles(files);
        const newFile = files.find((f: CsvFile) => f.id === csvFileId);
        if (newFile) {
          setSelectedCsvFile(newFile);
          setPage(1);
        }
      }

      toast({
        title: "Success",
        description: "CSV file processed successfully!",
        variant: "default",
      });
    } catch (error) {
      console.error("CSV processing error:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Error processing CSV file",
        variant: "destructive",
      });
    } finally {
      setIsProcessing(false);
    }
  };

  const handleTrainModel = async () => {
    if (!selectedCsvFile || !selectedProject) return;
    
    try {
      const labelsResponse = await fetch(
        `/api/projects/${selectedProject.id}/csv-files/${selectedCsvFile.id}/class-labels`
      );
      if (!labelsResponse.ok) {
        throw new Error('Failed to fetch class labels');
      }
      const labelsData = await labelsResponse.json();
      const labels = labelsData.map((label: any) => label.label);
  
      const rulesResponse = await fetch(
        `/api/projects/${selectedProject.id}/csv-files/${selectedCsvFile.id}/rules`
      );
      if (!rulesResponse.ok) {
        throw new Error('Failed to fetch rules');
      }
      const { rules } = await rulesResponse.json();
  
      const estimateResponse = await fetch('/api/estimate-cost', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          projectId: selectedProject.id,
          columnName: 'text',
          fileId: selectedCsvFile.id,
          rules: rules,
          labels: labels,
        }),
      });
  
      console.log(estimateResponse);
      if (!estimateResponse.ok) {
        throw new Error('Failed to get cost estimate');
      }
      
      const result = await estimateResponse.json();
      console.log('Data:', result);
      setCostEstimate({
        totalTokens: result.total_tokens || 0,
        promptTokens: result.prompt_tokens || 0,
        completionTokens: result.completion_tokens || 0,
        estimatedCost: result.estimated_cost || 0,
        sampleCount: result.sample_size || 0,
      });
      setShowCostModal(true);
    } catch (error) {
      console.error("Error getting cost estimate:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Error getting cost estimate",
        variant: "destructive",
      });
    }
  };

  const handleConfirmTraining = async () => {
    if (!selectedCsvFile || !selectedProject) return;
    
    setIsProcessing(true);
    setShowCostModal(false);
    setShowProgress(true);
    // Reset the flag so that we can update CSV once LLM Labeling completes
    setLlmLabelingCompleted(false);
    
    try {
      // Start the pipeline with the actual backend
      const pipelineResponse = await fetch('/api/run-pipeline', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          projectId: selectedProject.id,
          columnName: 'text', // Using 'text' as the default column
          fileId: selectedCsvFile.id,
        }),
      });
  
      if (!pipelineResponse.ok) {
        throw new Error('Failed to start pipeline');
      }
  
      const { job_id: jobId } = await pipelineResponse.json();
      console.log('Pipeline started with job ID:', jobId);
  
      // Poll the pipeline status every 2 seconds
      const statusInterval = setInterval(async () => {
        try {
          const statusResponse = await fetch(`/api/pipeline-status/${jobId}`);
          if (!statusResponse.ok) {
            throw new Error('Failed to get pipeline status');
          }
          const status = await statusResponse.json();
  
          // Update the pipeline stages in state.
          // If a stage is already marked as completed, keep it.
          setPipelineStages((prevStages) => {
            return prevStages.map((stage) => {
              // If the stage is already completed, don't update it
              if (stage.status === "completed") return stage;
  
              // Get the updated stage info from the API response
              const newStage = status.stages[stage.name];
              if (!newStage) return stage;
  
              // If the new stage is now completed, update it to completed
              if (newStage.status === "completed") {
                return {
                  ...stage,
                  status: "completed",
                  progress: 100,
                };
              }
  
              // Otherwise, update with the current status and progress
              return {
                ...stage,
                status: newStage.status,
                progress: newStage.progress,
              };
            });
          });
  
          // When LLM Labeling is completed (and not already updated), refresh the CSV data.
          if (
            status.stages &&
            status.stages["LLM Labeling"]?.status === "completed" &&
            !llmLabelingCompleted
          ) {
            console.log("LLM Labeling completed. Updating CSV data...");
            setLlmLabelingCompleted(true); // Prevent multiple reloads
            await loadCsvData();
          }
  
          // Stop polling when the overall pipeline job is finished.
          if (status.status === "completed" || status.status === "failed") {
            clearInterval(statusInterval);
            setIsProcessing(false);
            // setShowProgress(false);
            
            if (status.status === "completed") {
              toast({
                title: "Success",
                description: "Model training completed successfully!",
                variant: "default",
              });
            } else {
              toast({
                title: "Error", 
                description: status.error || "Model training failed",
                variant: "destructive",
              });
            }
          }
        } catch (error) {
          console.error("Error polling pipeline status:", error);
          clearInterval(statusInterval);
          setIsProcessing(false);
          // setShowProgress(false);
          toast({
            title: "Error",
            description: "Failed to get pipeline status",
            variant: "destructive",
          });
        }
      }, 1200);
    } catch (error) {
      console.error("Pipeline error:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Error running pipeline",
        variant: "destructive",
      });
      setIsProcessing(false);
      // setShowProgress(false);
    }
  };
  
  
  

  const handleAddProject = async () => {
    const name = prompt("Enter project name");
    if (name) {
      try {
        const response = await fetch('/api/projects', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ name }),
        });

        if (!response.ok) {
          throw new Error('Failed to create project');
        }

        const newProject = await response.json();
        setProjects(prev => [...prev, newProject]);
        toast({
          title: "Success",
          description: "Project created successfully",
          variant: "default",
        });
      } catch (error) {
        console.error("Error creating project:", error);
        toast({
          title: "Error",
          description: "Failed to create project",
          variant: "destructive",
        });
      }
    }
  }

  // Load class labels when CSV file is selected
  useEffect(() => {
    if (selectedCsvFile) {
      fetch(`/api/projects/${selectedProject?.id}/csv-files/${selectedCsvFile.id}/class-labels`)
        .then(res => res.json())
        .then(data => {
          if (Array.isArray(data)) {
            setClasses(data.map(item => item.label));
          }
        })
        .catch(error => {
          console.error('Error loading class labels:', error);
          toast({
            title: "Error",
            description: "Failed to load class labels",
            variant: "destructive",
          });
        });

      // Load classification rules
      fetch(`/api/projects/${selectedProject?.id}/csv-files/${selectedCsvFile.id}/rules`)
        .then(res => res.json())
        .then(data => {
          setRules(data.rules || "");
          setRulesLoaded(true);
        })
        .catch(error => {
          console.error('Error loading classification rules:', error);
          toast({
            title: "Error",
            description: "Failed to load classification rules",
            variant: "destructive",
          });
        });
    } else {
      setClasses([]);
      setRules("");
      setRulesLoaded(false);
    }
  }, [selectedCsvFile, selectedProject]);

  const addClass = async () => {
    if (!selectedCsvFile || !newClass || classes.includes(newClass)) return;

    try {
      const response = await fetch(`/api/projects/${selectedProject?.id}/csv-files/${selectedCsvFile.id}/class-labels`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label: newClass }),
      });

      if (!response.ok) {
        throw new Error('Failed to add class label');
      }

      setClasses(prev => [...prev, newClass]);
      setNewClass("");
      toast({
        title: "Success",
        description: "Class label added successfully",
        variant: "default",
      });
    } catch (error) {
      console.error('Error adding class label:', error);
      toast({
        title: "Error",
        description: "Failed to add class label",
        variant: "destructive",
      });
    }
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
  };

  const handleLabelChange = async (rowId: number, newLabel: string) => {
    if (!selectedProject) return;
    
    try {
      const response = await fetch(`/api/projects/${selectedProject.id}/csv-data`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          rowId: rowId,
          labelType: 'human_label',
          value: newLabel,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to update label');
      }

      setCsvData((prev) => {
        const index = prev.findIndex(row => row.id === rowId);
        if (index === -1) return prev;
        
        const updated = [...prev];
        updated[index] = { ...updated[index], human_label: newLabel };
        return updated;
      });
    } catch (error) {
      console.error("Error updating label:", error);
      toast({
        title: "Error",
        description: "Failed to update label",
        variant: "destructive",
      });
    }
  };

  const handleRulesChange = async (newRules: string) => {
    if (!selectedCsvFile || !rulesLoaded) return;
    
    setRules(newRules);
    
    try {
      await fetch(`/api/projects/${selectedProject?.id}/csv-files/${selectedCsvFile.id}/rules`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rules: newRules }),
      });
    } catch (error) {
      console.error('Error saving classification rules:', error);
      toast({
        title: "Error",
        description: "Failed to save classification rules",
        variant: "destructive",
      });
    }
  };

  return (
    <div className="h-[130vh] bg-background">
      <ToastProvider />

      <CostEstimateModal
        isOpen={showCostModal}
        onClose={() => setShowCostModal(false)}
        onConfirm={handleConfirmTraining}
        estimate={costEstimate}
        isLoading={isProcessing}
      />

      <nav className="border-b">
        <div className="container mx-auto flex h-16 items-center px-4">
          <div className="flex items-center gap-6">
            <h1 className="text-xl font-bold">Project Information</h1>
            <Link 
              href="/metrics" 
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              Metrics
            </Link>
          </div>
          <div className="ml-auto flex items-center gap-4">
            <ProjectSelect 
              projects={projects}
              selectedProject={selectedProject}
              onSelect={setSelectedProject}
            />
            <Button onClick={handleAddProject}>New Project</Button>
          </div>
        </div>
      </nav>

      <main className="container mx-auto py-6 px-4 h-[calc(100vh-4rem)]">
        <div className="grid grid-cols-3 gap-6">
          <div className="col-span-2 space-y-4">
            <DataTable 
              data={csvData} 
              onHover={setHoveredRowIndex}
              classes={classes}
              onLabelChange={handleLabelChange}
              page={page}
              totalPages={totalPages}
              onPageChange={setPage}
              pageSize={pageSize}
              onPageSizeChange={setPageSize}
            />
            
            <PipelineProgress
              stages={pipelineStages}
              isVisible={showProgress}
            />
          </div>

          <div className="space-y-6">
            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Upload File</h3>
              {!selectedProject ? (
                <p className="text-muted-foreground">Please select a project first</p>
              ) : (
                <>
                  <div className="space-y-4">
                    <div>
                      <label className="block font-medium mb-2">
                        Select CSV File
                      </label>
                      <CsvFileSelect
                        files={csvFiles}
                        selectedFile={selectedCsvFile}
                        onSelect={(file) => {
                          setSelectedCsvFile(file);
                          setPage(1); // Reset to first page when changing files
                        }}
                      />
                    </div>
                    <div>
                      <label className="block font-medium mb-2">
                        Upload New CSV
                      </label>
                      <Input
                        type="file"
                        accept=".csv"
                        onChange={handleFileUpload}
                        className="mb-4"
                        disabled={isProcessing}
                      />
                      <div className="flex flex-col gap-2">
                        <Button 
                          className="w-full" 
                          disabled={!selectedFile || isProcessing}
                          onClick={handleProcessCsv}
                        >
                          {isProcessing ? "Processing..." : "Process CSV"}
                        </Button>
                        <Button
                          className="w-full"
                          variant="secondary"
                          disabled={!selectedCsvFile || isProcessing}
                          onClick={handleTrainModel}
                        >
                          Train Model
                        </Button>
                      </div>
                    </div>
                  </div>
                </>
              )}
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Details</h3>
              {hoveredRowIndex !== null && csvData[hoveredRowIndex] ? (
                <div className="space-y-4">
                  <Textarea 
                    value={csvData[hoveredRowIndex].text} 
                    readOnly 
                    className="min-h-[100px]"
                  />
                  <div>
                    <h4 className="font-medium mb-2">Labels</h4>
                    <div className="flex flex-wrap gap-2">
                      {Object.entries(csvData[hoveredRowIndex])
                        .filter(([key]) => key.includes('label'))
                        .map(([key, value]) => (
                          <Badge key={key} variant="outline">
                            {key}: {value || 'N/A'}
                          </Badge>
                        ))
                      }
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">
                  Hover over a row to see details
                </p>
              )}
            </Card>

            <Card className="p-6">
              <h3 className="text-lg font-semibold mb-4">Configuration</h3>
              <div className="space-y-6">
                <div>
                  <label className="block font-medium mb-2">
                    Classification Rules
                  </label>
                  <Textarea
                    value={rules}
                    onChange={(e) => handleRulesChange(e.target.value)}
                    placeholder="Enter classification rules..."
                    className="min-h-[100px]"
                    disabled={!selectedCsvFile || !rulesLoaded}
                  />
                </div>

                <div>
                  <label className="block font-medium mb-2">
                    Class Labels
                  </label>
                  <div className="flex flex-wrap gap-2 mb-4">
                    {classes.map((cls) => (
                      <Badge key={cls}>{cls}</Badge>
                    ))}
                  </div>
                  <div className="flex gap-2">
                    <Input
                      value={newClass}
                      onChange={(e) => setNewClass(e.target.value)}
                      placeholder="Add new class..."
                    />
                    <Button onClick={addClass}>Add</Button>
                  </div>
                </div>
              </div>
            </Card>
          </div>
        </div>
      </main>
    </div>
  );
}
