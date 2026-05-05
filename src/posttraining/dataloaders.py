from torch.utils.data import DataLoader

from src.posttraining.dataset import EVERYDAY_TRAIN_SPLIT
from src.posttraining.dataset import EVERYDAY_VALIDATION_SPLIT
from src.posttraining.dataset import EverydayChatDataset
from src.posttraining.dataset import MagpieChatDataset
from src.tokenizer.tokenizer import ByteLevelBPE


def build_dataloaders(
    tokenizer: ByteLevelBPE,
    max_len: int,
    batch_size: int,
    num_workers: int,
    accelerator: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    # ---------------------------------------------------------
    # Resolve tokenizer ids shared by both SFT datasets and the
    # Transformer loss masking convention.
    # ---------------------------------------------------------
    pad_token_id = tokenizer.token_to_id(tokenizer.pad_token)
    bos_token_id = tokenizer.token_to_id(tokenizer.bos_token)
    eos_token_id = tokenizer.token_to_id(tokenizer.eos_token)
    end_of_turn_token_id = tokenizer.token_to_id(tokenizer.end_of_turn_token)

    # ---------------------------------------------------------
    # Build broad Magpie training, high-quality Everyday finishing,
    # and fixed Everyday validation datasets.
    # ---------------------------------------------------------
    magpie_dataset = MagpieChatDataset(
        tokenizer=tokenizer,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    everyday_train_dataset = EverydayChatDataset(
        tokenizer=tokenizer,
        split=EVERYDAY_TRAIN_SPLIT,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    everyday_validation_dataset = EverydayChatDataset(
        tokenizer=tokenizer,
        split=EVERYDAY_VALIDATION_SPLIT,
        max_len=max_len,
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        end_of_turn_token_id=end_of_turn_token_id,
    )
    use_pin_memory = accelerator == "cuda"
    use_persistent_workers = num_workers > 0

    # ---------------------------------------------------------
    # Wrap datasets with DataLoaders configured consistently with
    # the existing pretraining pipeline.
    # ---------------------------------------------------------
    magpie_dataloader = DataLoader(
        magpie_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    everyday_train_dataloader = DataLoader(
        everyday_train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    everyday_validation_dataloader = DataLoader(
        everyday_validation_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=use_persistent_workers,
    )
    return magpie_dataloader, everyday_train_dataloader, everyday_validation_dataloader
