  import Database from 'better-sqlite3';
  import path from 'path';

  // Assuming the shared database is one folder up in shared_data:
  const DB_PATH = path.join(process.cwd(), '../database/data.db');

  let db: Database.Database;

  export function getDb() {
    if (!db) {
      db = new Database(DB_PATH);
      initDb();
    }
    return db;
  }

  function initDb() {
    const db = getDb();
    
    // Enable foreign keys
    db.exec('PRAGMA foreign_keys = ON;');

    // Create tables if they don't exist
    db.exec(`
      CREATE TABLE IF NOT EXISTS projects (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      );
    `);

    // Create csv_files table
    db.exec(`
      CREATE TABLE IF NOT EXISTS csv_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id TEXT NOT NULL,
        filename TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (project_id) REFERENCES projects(id)
      );
    `);

    // Create csv_data table
    db.exec(`
      CREATE TABLE IF NOT EXISTS csv_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        csv_file_id INTEGER NOT NULL,
        text TEXT NOT NULL,
        llm_label TEXT,
        model_label TEXT,
        human_label TEXT,
        confidence FLOAT DEFAULT NULL,
        final_label TEXT,
        FOREIGN KEY (csv_file_id) REFERENCES csv_files(id)
      );
    `);

    // Create class_labels table
    db.exec(`
      CREATE TABLE IF NOT EXISTS class_labels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        csv_file_id INTEGER NOT NULL,
        label TEXT NOT NULL,
        FOREIGN KEY (csv_file_id) REFERENCES csv_files(id)
      );
    `);

    // Create classification_rules table
    db.exec(`
      CREATE TABLE IF NOT EXISTS classification_rules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        csv_file_id INTEGER NOT NULL UNIQUE,
        rules TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (csv_file_id) REFERENCES csv_files(id)
      );
    `);
  }

  // Project operations
  export function createProject(name: string) {
    const db = getDb();
    const id = Date.now().toString();
    db.prepare('INSERT INTO projects (id, name) VALUES (?, ?)').run(id, name);
    return { id, name };
  }

  export function getProjects() {
    const db = getDb();
    return db.prepare('SELECT * FROM projects ORDER BY created_at DESC').all();
  }

  export function getProject(id: string) {
    const db = getDb();
    return db.prepare('SELECT * FROM projects WHERE id = ?').get(id);
  }

  // CSV file operations
  export function createCsvFile(projectId: string, filename: string) {
    const db = getDb();
    const result = db.prepare('INSERT INTO csv_files (project_id, filename) VALUES (?, ?)').run(projectId, filename);
    return result.lastInsertRowid;
  }

  export function getCsvFiles(projectId: string) {
    const db = getDb();
    return db.prepare('SELECT * FROM csv_files WHERE project_id = ? ORDER BY created_at DESC').all(projectId);
  }

  // CSV data operations
  export function insertCsvData(csvFileId: number, data: { text: string; llm_label?: string; model_label?: string; human_label?: string; final_label?: string }[]) {
    const db = getDb();
    const stmt = db.prepare(`
      INSERT INTO csv_data (csv_file_id, text, llm_label, model_label, human_label, final_label)
      VALUES (?, ?, ?, ?, ?, ?)
    `);

    const insertMany = db.transaction((rows: { text: string; llm_label?: string; model_label?: string; human_label?: string; final_label?: string }[]) => {
      for (const row of rows) {
        stmt.run(
          csvFileId,
          row.text,
          row.llm_label || null,
          row.model_label || null,
          row.human_label || null,
          row.final_label || null
        );
      }
    });

    insertMany(data);
  }

  interface CsvDataRow {
    id: number;
    csv_file_id: number;
    text: string;
    llm_label: string | null;
    model_label: string | null;
    human_label: string | null;
    final_label: string | null;
  }

  interface CountResult {
    count: number;
  }

  export function getCsvData(csvFileId: number, page: number = 1, pageSize: number = 20) {
    const db = getDb();
    const offset = (page - 1) * pageSize;
    
    const data = db.prepare('SELECT * FROM csv_data WHERE csv_file_id = ? LIMIT ? OFFSET ?').all(csvFileId, pageSize, offset) as CsvDataRow[];
    const total = (db.prepare('SELECT COUNT(*) as count FROM csv_data WHERE csv_file_id = ?').get(csvFileId) as CountResult).count;
    
    return {
      data,
      total,
      page,
      pageSize,
      totalPages: Math.ceil(total / pageSize)
    };
  }

  export function updateLabel(id: number, labelType: 'human_label' | 'llm_label' | 'model_label' | 'final_label', value: string) {
    const db = getDb();
    db.prepare(`UPDATE csv_data SET ${labelType} = ? WHERE id = ?`).run(value, id);
  }

  // Class labels operations
  export function addClassLabel(csvFileId: number, label: string) {
    const db = getDb();
    db.prepare('INSERT INTO class_labels (csv_file_id, label) VALUES (?, ?)').run(csvFileId, label);
  }

  export function getClassLabels(csvFileId: number) {
    const db = getDb();
    return db.prepare('SELECT * FROM class_labels WHERE csv_file_id = ?').all(csvFileId);
  }

  export function setClassificationRules(csvFileId: number, rules: string) {
    const db = getDb();
    const stmt = db.prepare(`
      INSERT INTO classification_rules (csv_file_id, rules)
      VALUES (?, ?)
      ON CONFLICT(csv_file_id) DO UPDATE SET rules = excluded.rules
    `);
    stmt.run(csvFileId, rules);
  }

  export function getClassificationRules(csvFileId: number): { rules: string } | undefined {
    const db = getDb();
    return db.prepare('SELECT rules FROM classification_rules WHERE csv_file_id = ?').get(csvFileId) as { rules: string } | undefined;
  }
