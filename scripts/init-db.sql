-- Create the mlflow database alongside the main app database.
-- This runs once on first PostgreSQL container boot.
SELECT 'CREATE DATABASE mlflow OWNER appuser'
  WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow')\gexec
