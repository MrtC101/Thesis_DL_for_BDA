import logging

def get_logger(name: str, level: int) -> logging.Logger:
        """
        Configura y devuelve un objeto Logger.

        Args:
            name (str): Nombre del logger.
            level (int): Nivel de registro. Utiliza los niveles de registro de la enumeración de módulo logging.

        Returns:
            logging.Logger: Objeto Logger configurado.
        """
        # Crea un objeto Logger con el nombre proporcionado
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Configura el nivel de registro del logger a DEBUG

        # Define el formato del registro
        formatter = logging.Formatter(
            '%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
            datefmt='%m-%d-%Y %H:%M'
        )

        # Configura un manejador de flujo (stream handler) para la salida de registro
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(level)  # Configura el nivel de registro del manejador de flujo
        stream_handler.setFormatter(formatter)  # Aplica el formato definido al manejador de flujo

        # Agrega el manejador de flujo al logger
        logger.addHandler(stream_handler)

        return logger