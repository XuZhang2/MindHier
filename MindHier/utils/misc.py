import datetime
import functools
import glob
import os
import subprocess
import sys
import time
from collections import deque
from typing import List, Tuple

import numpy as np
import pytz
import torch
import torch.distributed as tdist

import dist
import wandb

os_system = functools.partial(subprocess.call, shell=True)


def echo(info):
    os_system(
        f'echo "[$(date "+%m-%d-%H:%M:%S")] ({os.path.basename(sys._getframe().f_back.f_code.co_filename)}, line{sys._getframe().f_back.f_lineno})=> {info}"'
    )


def os_system_get_stdout(cmd):
    return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode(
        "utf-8"
    )


def os_system_get_stdout_stderr(cmd):
    cnt = 0
    while True:
        try:
            sp = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            cnt += 1
            print(f"[fetch free_port file] timeout cnt={cnt}")
        else:
            return sp.stdout.decode("utf-8"), sp.stderr.decode("utf-8")


def time_str(fmt="[%m-%d %H:%M:%S]"):
    return datetime.datetime.now(tz=pytz.timezone("Asia/Shanghai")).strftime(fmt)


def init_distributed_mode(local_out_path, only_sync_master=False):
    try:
        dist.initialize()
        dist.barrier()
    except RuntimeError:
        print(f'{">"*75}  NCCL Error  {"<"*75}', flush=True)
        time.sleep(10)
    if local_out_path is not None:
        os.makedirs(local_out_path, exist_ok=True)
    _change_builtin_print(dist.is_local_master())
    if (
        (dist.is_master() if only_sync_master else dist.is_local_master())
        and local_out_path is not None
        and len(local_out_path)
    ):
        sys.stdout, sys.stderr = SyncPrint(local_out_path, sync_stdout=True), SyncPrint(
            local_out_path, sync_stdout=False
        )


def _change_builtin_print(is_master):
    import builtins as __builtin__

    builtin_print = __builtin__.print
    if type(builtin_print) != type(open):
        return

    def prt(*args, **kwargs):
        force = kwargs.pop("force", False)
        clean = kwargs.pop("clean", False)
        deeper = kwargs.pop("deeper", False)
        if is_master or force:
            if not clean:
                f_back = sys._getframe().f_back
                if deeper and f_back.f_back is not None:
                    f_back = f_back.f_back
                file_desc = f"{f_back.f_code.co_filename:24s}"[-24:]
                builtin_print(
                    f"{time_str()} ({file_desc}, line{f_back.f_lineno:-4d})=>",
                    *args,
                    **kwargs,
                )
            else:
                builtin_print(*args, **kwargs)

    __builtin__.print = prt


class SyncPrint(object):
    def __init__(self, local_output_dir, sync_stdout=True):
        self.sync_stdout = sync_stdout
        self.terminal_stream = sys.stdout if sync_stdout else sys.stderr
        fname = os.path.join(
            local_output_dir, "stdout.txt" if sync_stdout else "stderr.txt"
        )
        existing = os.path.exists(fname)
        self.file_stream = open(fname, "a")
        if existing:
            self.file_stream.write(
                "\n" * 7 + "=" * 55 + f"   RESTART {time_str()}   " + "=" * 55 + "\n"
            )
        self.file_stream.flush()
        self.enabled = True

    def write(self, message):
        self.terminal_stream.write(message)
        self.file_stream.write(message)

    def flush(self):
        self.terminal_stream.flush()
        self.file_stream.flush()

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        self.file_stream.flush()
        self.file_stream.close()
        if self.sync_stdout:
            sys.stdout = self.terminal_stream
            sys.stdout.flush()
        else:
            sys.stderr = self.terminal_stream
            sys.stderr.flush()

    def __del__(self):
        self.close()


class DistLogger(object):
    def __init__(self, lg, verbose):
        self._lg, self._verbose = lg, verbose

    @staticmethod
    def do_nothing(*args, **kwargs):
        pass

    def __getattr__(self, attr: str):
        return getattr(self._lg, attr) if self._verbose else DistLogger.do_nothing


class TensorboardLogger(object):
    def __init__(self, log_dir, filename_suffix):
        try:
            import tensorflow_io as tfio
        except:
            pass
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(log_dir=log_dir, filename_suffix=filename_suffix)
        self.step = 0

    def set_step(self, step=None):
        if step is not None:
            self.step = step
        else:
            self.step += 1

    def update(self, head="scalar", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            # assert isinstance(v, (float, int)), type(v)
            if step is None:  # iter wise
                it = self.step
                if it == 0 or (it + 1) % 500 == 0:
                    if hasattr(v, "item"):
                        v = v.item()
                    self.writer.add_scalar(f"{head}/{k}", v, it)
            else:  # epoch wise
                if hasattr(v, "item"):
                    v = v.item()
                self.writer.add_scalar(f"{head}/{k}", v, step)

    def update_logits(self, head="hist", step=None, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            # assert isinstance(v, (float, int)), type(v)
            self.log_tensor_as_distri(f"{head}/{k}", v, step)

    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        if step is None:  # iter wise
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:  # epoch wise
            loggable = True
        if loggable:
            try:
                self.writer.add_histogram(tag=tag, values=tensor1d, global_step=step)
            except Exception as e:
                print(f"[log_tensor_as_distri writer.add_histogram failed]: {e}")

    def log_text(self, tag, text, step=None):
        if step is None:  # iter wise
            step = self.step
            loggable = step == 0 or (step + 1) % 500 == 0
        else:  # epoch wise
            loggable = True
        if loggable:
            self.writer.add_text(tag, text, step)

    def log_image(self, tag, img_chw, step=None):
        self.writer.add_image(tag, img_chw, step, dataformats="CHW")

    def flush(self):
        self.writer.flush()

    def close(self):
        self.writer.close()

class WandbLogger(object):
    def __init__(self):
        """
        Initialize WandbLogger 
        """
        import wandb
        self.wandb = wandb
        if not wandb.run:
            print("Warning: No active wandb run detected. Make sure to call wandb.init() before creating WandbLogger.")
        self.step = 0
    
    def set_step(self, step=None):
        """
        Set the current step or increment if step is None
        
        Args:
            step: If provided, sets the step to this value; otherwise increments internal step
        """
        if step is not None:
            self.step = step
        else:
            self.step += 1
    
    def update(self, head="scalar", step=None, **kwargs):
        """
        Log scalar values to wandb
        
        Args:
            head: Category prefix for metrics
            step: Step value to use, defaults to current step
            **kwargs: Key-value pairs to log
        """
        for k, v in kwargs.items():
            if v is None:
                continue
                
            # Handle tensor values
            if hasattr(v, "item"):
                v = v.item()
                
            current_step = step if step is not None else self.step
            
            # Match TensorboardLogger's behavior of logging only at specific steps
            if step is None:  # iter wise
                if current_step == 0 or (current_step + 1) % 500 == 0:
                    self.wandb.log({f"{head}/{k}": v}, step=current_step)
            else:  # epoch wise
                self.wandb.log({f"{head}/{k}": v}, step=current_step)
    
    def update_logits(self, head="hist", step=None, **kwargs):
        """
        Log tensor distributions to wandb
        
        Args:
            head: Category prefix for histograms
            step: Step value to use, defaults to current step
            **kwargs: Key-value pairs of tensors to log as histograms
        """
        for k, v in kwargs.items():
            if v is None:
                continue
            self.log_tensor_as_distri(f"{head}/{k}", v, step)
    
    def log_tensor_as_distri(self, tag, tensor1d, step=None):
        """
        Log a tensor as a histogram distribution
        
        Args:
            tag: Name of the histogram
            tensor1d: Tensor values to log
            step: Step value to use, defaults to current step
        """
        current_step = step if step is not None else self.step
        
        # Match TensorboardLogger's behavior of logging only at specific steps
        loggable = True
        if step is None:  # iter wise
            loggable = current_step == 0 or (current_step + 1) % 500 == 0
            
        if loggable:
            try:
                import numpy as np
                if hasattr(tensor1d, "detach"):
                    tensor1d = tensor1d.detach()
                if hasattr(tensor1d, "cpu"):
                    tensor1d = tensor1d.cpu()
                if hasattr(tensor1d, "numpy"):
                    tensor1d = tensor1d.numpy()
                
                # Log as histogram
                self.wandb.log({tag: self.wandb.Histogram(tensor1d)}, step=current_step)
            except Exception as e:
                print(f"[log_tensor_as_distri wandb.log histogram failed]: {e}")
    
    def log_text(self, tag, text, step=None):
        """
        Log text to wandb
        
        Args:
            tag: Name for the text entry
            text: Text content to log
            step: Step value to use, defaults to current step
        """
        current_step = step if step is not None else self.step
        
        # Match TensorboardLogger's behavior of logging only at specific steps
        loggable = True
        if step is None:  # iter wise
            loggable = current_step == 0 or (current_step + 1) % 500 == 0
            
        if loggable:
            self.wandb.log({tag: self.wandb.Html(f"<pre>{text}</pre>")}, step=current_step)
    
    def log_image(self, tag, img_chw, step=None):
        """
        Log an image to wandb
        
        Args:
            tag: Name for the image
            img_chw: Image tensor in CHW format
            step: Step value to use, defaults to current step
        """
        current_step = step if step is not None else self.step
        
        # Convert tensor to numpy if needed
        if hasattr(img_chw, "detach"):
            img_chw = img_chw.detach()
        if hasattr(img_chw, "cpu"):
            img_chw = img_chw.cpu()
        if hasattr(img_chw, "numpy"):
            img_chw = img_chw.numpy()
            
        # wandb expects HWC format, so transpose from CHW
        import numpy as np
        img_hwc = np.transpose(img_chw, (1, 2, 0))
        
        self.wandb.log({tag: self.wandb.Image(img_hwc)}, step=current_step)
    
    def flush(self):
        """
        No-op for API compatibility with TensorboardLogger
        """
        pass  # Wandb flushes automatically
    
    def close(self):
        """
        Close the wandb run
        """
        if self.wandb.run:
            try:
                # Handle the case where sys.stderr might not have isatty()
                self.wandb.finish(quiet=True)
            except AttributeError as e:
                if "'SyncPrint' object has no attribute 'isatty'" in str(e):
                    # Fallback approach when isatty is not available
                    import sys
                    original_stderr = sys.stderr
                    try:
                        # Replace stderr temporarily with something that has isatty
                        import io
                        class FakeStderr(io.StringIO):
                            def isatty(self):
                                return False
                        
                        sys.stderr = FakeStderr()
                        self.wandb.finish(quiet=True)
                    finally:
                        # Restore original stderr
                        sys.stderr = original_stderr

class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=30, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not tdist.is_available() or not tdist.is_initialized():
            return
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        tdist.barrier()
        tdist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return np.median(self.deque) if len(self.deque) else 0

    @property
    def avg(self):
        return sum(self.deque) / (len(self.deque) or 1)

    @property
    def global_avg(self):
        return self.total / (self.count or 1)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1] if len(self.deque) else 0

    def time_preds(self, counts) -> Tuple[float, str, str]:
        remain_secs = counts * self.median
        return (
            remain_secs,
            str(datetime.timedelta(seconds=round(remain_secs))),
            time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + remain_secs)),
        )

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


def glob_with_latest_modified_first(pattern, recursive=False):
    return sorted(
        glob.glob(pattern, recursive=recursive), key=os.path.getmtime, reverse=True
    )


def auto_resume(
    args, pattern="ckpt*.pth"
) -> Tuple[List[str], int, dict, dict]:
    info = []
    file = os.path.join(args.local_out_dir_path, pattern)
    all_ckpt = glob_with_latest_modified_first(file)
    if len(all_ckpt) == 0:
        info.append(f"[auto_resume] no ckpt found @ {file}")
        info.append(f"[auto_resume quit]")
        return info, 0, {}, {}
    else:
        info.append(f"[auto_resume] load ckpt from @ {all_ckpt[0]} ...")
        ckpt = torch.load(all_ckpt[0], map_location="cpu")
        it = ckpt["iter"]
        info.append(f"[auto_resume success] resume from iteration {it}")
        return info, it, ckpt["trainer"], ckpt["args"]
