import os
import uvicorn
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    try:
        port = int(os.getenv("PORT", 8000))
        
        logger.info(f"Starting ML Training API on port {port}")
        
        # Run the FastAPI application with uvicorn
        uvicorn.run(
            "app.main:app", 
            host="0.0.0.0", 
            port=port,
            reload=True  # Enable hot-reloading for development
        )
    except Exception as e:
        logger.error(f"Error starting ML Training API: {str(e)}")
        raise 