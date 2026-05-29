┌──────────────────────────────┐
│ Start train_model_simple()   │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Initialize:                  │
│ train_losses = []            │
│ val_losses = []              │
│ track_tokens_seen = []       │
│ tokens_seen = 0              │
│ global_step = -1             │
└──────────────┬───────────────┘
               │
               ▼
       ┌─────────────────┐
       │ For each epoch  │
       │ in num_epochs   │
       └───────┬─────────┘
               │
               ▼
┌──────────────────────────────┐
│ model.train()                │
│ Set training mode            │
└──────────────┬───────────────┘
               │
               ▼
      ┌───────────────────────┐
      │ For each batch in     │
      │ train_loader          │
      └──────────┬────────────┘
                 │
                 ▼
┌──────────────────────────────┐
│ optimizer.zero_grad()        │
│ Clear old gradients          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ loss = calc_loss_batch()     │
│ Forward pass + loss          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ loss.backward()              │
│ Compute gradients            │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ optimizer.step()             │
│ Update weights               │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ tokens_seen +=               │
│ input_batch.numel()          │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ global_step += 1             │
└──────────────┬───────────────┘
               │
               ▼
        ┌─────────────────┐
        │ global_step %   │
        │ eval_freq == 0? │
        └───────┬─────────┘
            Yes │ No
                │
                ▼
     ┌───────────────────────┐
     │ Continue training     │
     │ next batch            │
     └───────────────────────┘

Yes branch:
     ▼
┌──────────────────────────────┐
│ evaluate_model()             │
│ Compute train_loss           │
│ Compute val_loss             │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Append losses to lists       │
│ Append tokens_seen           │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Print progress               │
│ Train loss / Val loss        │
└──────────────┬───────────────┘
               │
               ▼
      (Back to next batch)

─────────────────────────────────
After all batches in an epoch:
─────────────────────────────────

               ▼
┌──────────────────────────────┐
│ generate_and_print_sample()  │
│ Generate sample text         │
└──────────────┬───────────────┘
               │
               ▼
        ┌─────────────────┐
        │ More epochs?    │
        └───────┬─────────┘
            Yes │ No
                │
                ▼
      Back to epoch loop

No branch:
               ▼
┌──────────────────────────────┐
│ Return:                      │
│ train_losses                 │
│ val_losses                   │
│ track_tokens_seen            │
└──────────────┬───────────────┘
               │
               ▼
            End




===============================================

`Compact version`
Start
  │
  ▼
Initialize Variables
  │
  ▼
For Each Epoch
  │
  ├── Set Model to Training Mode
  │
  └── For Each Batch
         │
         ├── Zero Gradients
         ├── Forward Pass + Loss
         ├── Backward Pass
         ├── Update Weights
         ├── Count Tokens
         ├── Increment Step
         │
         └── Evaluation Step?
                │
                ├── Yes → Evaluate → Store Metrics → Print
                │
                └── No
  │
  └── Generate Sample Text
  │
  ▼
Return Loss History & Token Counts
  │
  ▼
End