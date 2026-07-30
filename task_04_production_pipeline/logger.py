import logging 
import os 
from logging.handlers import RotatingFileHandler

def setup_logger(name, log_file='logs/app.log', level=logging.INFO):  #Ye Filter hai. logging.DEBUG (10), INFO (20), WARNING (30), ERROR (40), CRITICAL (50).Agar level INFO set kiya, toh logger.debug("xyz") terminal par nahi dikhegi (kyunki DEBUG < INFO). Sirf INFO aur usse upar wali dikhegi.


    """
    Production-grade logger setup with rotating file handler.
    """
    # Step A: Logger instance banao
    logger = logging.getLogger(name) # ⭐ STAR: name me hum usually __name__ (current file ka naam) dete hain. Isse logs me pata chalta hai ki "ye log kis file se aa raha hai".
                                     #⭐ STAR: Python me logger Singleton (Ek hi instance) hota hai. Matlab agar tum 10 jagah getLogger("mylogger") likho, toh sab ek hi object use karte hain. Isse logs duplicate nahi hote.
    # Step B: Logging level set karo (threshold)
    logger.setLevel(level)

        # Step C: Formatting (Kaise dikhega log)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

        # Step D: Handlers set karo (Destination)
    if not logger.handlers:
        # 1. Console Handler (Terminal ke liye)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # 2. File Handler (Rotating - Production Grade)
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=3,         # Sirf 3 purani files rakho
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    # Logger ko call karo
    log = setup_logger('my_app', log_file='logs/test.log')
    
    log.debug("Ye DEBUG hai (Nahi dikhega kyunki level INFO hai)")
    log.info("Application Start ho gayi!")
    log.warning("Disk space kam ho raha hai!")
    log.error("Database connection fail ho gayi!")
    log.critical("Server crash ho raha hai!")
    
    print("\n✅ Logs check karo: 'logs/test.log' file mein aur terminal mein.")