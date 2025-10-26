"""Argument parsing utilities"""

def print_args(args, parser=None):
    """Print parsed arguments in a readable format"""
    print("\n" + "="*60)
    print("Configuration:")
    print("="*60)
    
    for arg, value in vars(args).items():
        print(f"  {arg:25s}: {value}")
    
    print("="*60 + "\n")
