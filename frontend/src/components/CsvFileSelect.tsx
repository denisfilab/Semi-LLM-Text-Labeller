"use client";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface CsvFile {
  id: number;
  filename: string;
  created_at: string;
}

interface CsvFileSelectProps {
  files: CsvFile[];
  selectedFile: CsvFile | null;
  onSelect: (file: CsvFile | null) => void;
}

export function CsvFileSelect({ files, selectedFile, onSelect }: CsvFileSelectProps) {
  const handleChange = (value: string) => {
    const selected = files.find(f => f.id.toString() === value) || null;
    onSelect(selected);
  };

  return (
    <Select onValueChange={handleChange} value={selectedFile?.id.toString()}>
      <SelectTrigger className="w-[200px]">
        <SelectValue placeholder="Select CSV file" />
      </SelectTrigger>
      <SelectContent>
        {files.map((file) => (
          <SelectItem key={file.id} value={file.id.toString()}>
            {file.filename}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
