from collections import defaultdict

def nested_defaultdict(depth, default_factory=dict):
    """
    Crea un defaultdict anidado con la profundidad especificada.
    
    Args:
        depth (int): Profundidad del anidamiento.
        default_factory (callable, opcional): Función para inicializar los defaultdicts. Por defecto es dict.
    
    Returns:
        defaultdict: defaultdict anidado con la profundidad especificada.
    """
    if depth <= 1:
        return defaultdict(default_factory)
    else:
        return defaultdict(lambda: nested_defaultdict(depth - 1, default_factory))
