-- This runs automatically when container first starts
-- We need to define vector type before SQLAlchemy tries to use it
CREATE EXTENSION IF NOT EXISTS vector;

-- Test it works
SELECT '[-1,2,3]'::vector;
