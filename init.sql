-- Ensure the database exists (optional, only needed if not created in docker-compose)
SELECT 'CREATE DATABASE mnist_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mnist_db')\gexec

-- Connect to the database
\c mnist_db;

-- Create the predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,  -- Automatically store the time of prediction
    predicted_digit INTEGER NOT NULL,              -- The model’s predicted digit
    true_label INTEGER                             -- The actual label (optional, can be NULL if not available)
);
