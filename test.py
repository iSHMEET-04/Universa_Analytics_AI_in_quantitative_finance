import torch
print(torch.cuda.is_available())         # Should print True
print(torch.cuda.device_count())         # Should print >0
print(torch.cuda.get_device_name(0))  