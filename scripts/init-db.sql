-- Initialize database for Reflexion Agent

-- Create database if not exists (handled by Docker environment)
-- CREATE DATABASE IF NOT EXISTS reflexion;

-- Create user and grant permissions (handled by Docker environment)
-- CREATE USER IF NOT EXISTS 'reflexion'@'%' IDENTIFIED BY 'reflexion_pass';
-- GRANT ALL PRIVILEGES ON reflexion.* TO 'reflexion'@'%';

-- Create tables for reflexion memory storage
CREATE TABLE IF NOT EXISTS reflections (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(255) NOT NULL,
    task_description TEXT NOT NULL,
    output TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    score DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    issues TEXT[] DEFAULT '{}',
    improvements TEXT[] DEFAULT '{}',
    confidence DECIMAL(3,2) NOT NULL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS episodes (
    id SERIAL PRIMARY KEY,
    task_description TEXT NOT NULL,
    final_output TEXT NOT NULL,
    success BOOLEAN NOT NULL DEFAULT FALSE,
    iterations INTEGER NOT NULL DEFAULT 1,
    total_time DECIMAL(10,3) NOT NULL DEFAULT 0.0,
    llm_model VARCHAR(100) NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patterns (
    id SERIAL PRIMARY KEY,
    pattern_type VARCHAR(50) NOT NULL,
    pattern_description TEXT NOT NULL,
    frequency INTEGER NOT NULL DEFAULT 1,
    success_correlation DECIMAL(3,2) DEFAULT 0.0,
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_reflections_task_id ON reflections(task_id);
CREATE INDEX IF NOT EXISTS idx_reflections_created_at ON reflections(created_at);
CREATE INDEX IF NOT EXISTS idx_reflections_success ON reflections(success);

CREATE INDEX IF NOT EXISTS idx_episodes_created_at ON episodes(created_at);
CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
CREATE INDEX IF NOT EXISTS idx_episodes_llm_model ON episodes(llm_model);

CREATE INDEX IF NOT EXISTS idx_patterns_type ON patterns(pattern_type);
CREATE INDEX IF NOT EXISTS idx_patterns_frequency ON patterns(frequency);

-- Insert sample data for testing
INSERT INTO episodes (task_description, final_output, success, iterations, total_time, llm_model) 
VALUES 
    ('Write hello world function', 'def hello(): return "Hello, World!"', true, 1, 1.5, 'gpt-4'),
    ('Implement binary search', 'def binary_search(arr, target): ...', true, 2, 3.2, 'gpt-4'),
    ('Create REST endpoint', 'from flask import Flask...', false, 3, 8.7, 'gpt-3.5-turbo')
ON CONFLICT DO NOTHING;

INSERT INTO reflections (task_id, task_description, output, success, score, issues, improvements)
VALUES 
    ('task-001', 'Debug segmentation fault', 'Check array bounds', false, 0.3, 
     ARRAY['Incomplete solution', 'Missing edge cases'], 
     ARRAY['Add bounds checking', 'Include error handling'])
ON CONFLICT DO NOTHING;

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create trigger for reflections table
DROP TRIGGER IF EXISTS update_reflections_updated_at ON reflections;
CREATE TRIGGER update_reflections_updated_at
    BEFORE UPDATE ON reflections
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();