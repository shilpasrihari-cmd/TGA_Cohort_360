"""Fail-closed local storage boundaries; never include paths or identities in errors."""
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import re
import stat
import subprocess
from .restricted_inventory import InventoryValidationError

def canonical_path(path):
    lexical = Path(os.path.abspath(path))
    for component in [*reversed(lexical.parents), lexical]:
        try:
            metadata = component.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise InventoryValidationError('Cannot verify physical storage isolation.') from None
        if stat.S_ISLNK(metadata.st_mode) or getattr(metadata,'st_file_attributes',0) & 0x400:
            raise InventoryValidationError('Unsafe symbolic link, junction, or reparse point in storage path.')
    try:
        resolved = lexical.resolve()
    except OSError:
        raise InventoryValidationError('Cannot canonicalize storage path.') from None
    if resolved != lexical:
        raise InventoryValidationError('Storage path resolves through an unsafe alias.')
    return resolved

def checked_path(path, root, folders=('private','staging')):
    root = canonical_path(root)
    resolved = canonical_path(path)
    allowed = [canonical_path(root/'data'/folder) for folder in folders]
    if not any(resolved.is_relative_to(base) for base in allowed):
        raise InventoryValidationError('Storage path escapes approved roots.')
    return resolved

def verify_mode_isolation(root):
    root = canonical_path(root)
    for folder in ['private','staging','public']:
        a = canonical_path(root/'data'/folder/'synthetic')
        b = canonical_path(root/'data'/folder/'real')
        if a == b:
            raise InventoryValidationError('Synthetic and real storage are physically aliased.')
        try:
            if a.exists() and b.exists() and os.path.samefile(a,b):
                raise InventoryValidationError('Synthetic and real storage are physically aliased.')
        except OSError:
            raise InventoryValidationError('Cannot verify physical mode isolation.') from None

def verify_no_overlap(input_path, outputs):
    source = canonical_path(input_path)
    try:
        if source.exists() and source.stat().st_nlink > 1:
            raise InventoryValidationError('Hard-linked restricted input is not permitted.')
        for output in outputs:
            target = canonical_path(output)
            if source == target or source.is_relative_to(target) or target.is_relative_to(source):
                raise InventoryValidationError('Input and output storage overlap.')
            if source.exists() and target.exists() and os.path.samefile(source,target):
                raise InventoryValidationError('Input and output storage are physically aliased.')
    except OSError:
        raise InventoryValidationError('Cannot verify input/output isolation.') from None

def restrict_permissions(path):
    """Owner-only POSIX access; protected owner + SYSTEM DACL on Windows."""
    path = canonical_path(path)
    try:
        if os.name != 'nt':
            path.chmod(0o700 if path.is_dir() else 0o600)
            return
        result = subprocess.run(['whoami','/user','/fo','csv','/nh'],capture_output=True,text=True,check=True)
        sid = re.search(r'S-1-5-(?:\d+-)*\d+',result.stdout)
        if not sid:
            raise OSError('No SID')
        advapi = ctypes.WinDLL('advapi32',use_last_error=True)
        kernel = ctypes.WinDLL('kernel32',use_last_error=True)
        convert = advapi.ConvertStringSecurityDescriptorToSecurityDescriptorW
        convert.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,ctypes.POINTER(ctypes.c_void_p),ctypes.c_void_p]
        convert.restype = wintypes.BOOL
        extract = advapi.GetSecurityDescriptorDacl
        extract.argtypes = [ctypes.c_void_p,ctypes.POINTER(wintypes.BOOL),ctypes.POINTER(ctypes.c_void_p),ctypes.POINTER(wintypes.BOOL)]
        extract.restype = wintypes.BOOL
        apply = advapi.SetNamedSecurityInfoW
        apply.argtypes = [wintypes.LPWSTR,wintypes.DWORD,wintypes.DWORD,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        apply.restype = wintypes.DWORD
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        descriptor = ctypes.c_void_p()
        flags = 'OICI' if path.is_dir() else ''
        sddl = f'D:P(A;{flags};FA;;;{sid.group()})(A;{flags};FA;;;SY)'
        if not convert(sddl,1,ctypes.byref(descriptor),None):
            raise OSError('DACL conversion failed')
        try:
            present,defaulted,dacl = wintypes.BOOL(),wintypes.BOOL(),ctypes.c_void_p()
            if not extract(descriptor,ctypes.byref(present),ctypes.byref(dacl),ctypes.byref(defaulted)) or not present.value:
                raise OSError('DACL extraction failed')
            if apply(str(path),1,0x80000004,None,None,dacl,None)!=0:
                raise OSError('DACL application failed')
        finally:
            kernel.LocalFree(descriptor)
    except Exception:
        raise InventoryValidationError('Cannot enforce least-privilege storage permissions; processing stopped.') from None

def secure_directory(path, root, folders=('private','staging')):
    path = checked_path(path,root,folders)
    path.mkdir(parents=True,exist_ok=True,mode=0o700)
    checked_path(path,root,folders)
    restrict_permissions(path)
    return path

def secure_csv(frame, path, root):
    path = checked_path(path,root)
    with path.open('x',encoding='utf-8',newline='') as stream:
        frame.to_csv(stream,index=False)
    restrict_permissions(path)
