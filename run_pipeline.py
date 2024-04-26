import runpy
import os

if __name__ == "__main__":
    #
    args = ["arg1_value", "arg2_value"]
    mod = os.path("")
    runpy.run_module(mod, run_name="__main__", alter_sys=True, run_globals={"__name__": "__main__"})
    args = ["",]
    mod = os.path("")
    runpy.run_module(mod,init_globals=args, run_name="__main__")
    args = ["",]
    mod = os.path("")
    runpy.run_module(mod,init_globals=args, run_name="__main__")