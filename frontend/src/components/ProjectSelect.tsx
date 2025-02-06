import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
  } from "@/components/ui/select"
  
  interface Project {
    id: string;
    name: string;
    csvPath?: string;
    createdAt: Date;
  }
  
  interface ProjectSelectProps {
    projects: Project[];
    selectedProject: Project | null;
    onSelect: (project: Project | null) => void;
  }
  
  
  export function ProjectSelect({ projects, selectedProject, onSelect }: ProjectSelectProps) {
    // Handle the string to Project conversion here
    const handleChange = (value: string) => {
      const selected = projects.find(p => p.id === value) || null;
      onSelect(selected);
    };
    return (
      <Select onValueChange={handleChange} value={selectedProject?.id}>
        <SelectTrigger className="w-[180px]">
          <SelectValue placeholder="Select project" />
        </SelectTrigger>
        <SelectContent>
          {projects.map((project) => (
            <SelectItem key={project.id} value={project.id}>
              {project.name}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    )
  }