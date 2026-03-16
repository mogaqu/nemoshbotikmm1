@echo off
set "CANDY_WEIGHTS=weights/balls+coins.pt"
set "PLAYER_WEIGHTS=weights/players_v1.pt"
set "CONFIDENCE=0.22"
set "PLAYER_CONFIDENCE=0.45"
set "CANDY_FRAME_SKIP=2"
set "PLAYER_FRAME_SKIP=6"

echo [INFO] Starting Predator Bot v5.4 (Ghost Hunter Edition)...
echo [INFO] Using PyTorch models for GPU compatibility!
echo [INFO] Candy Model: %CANDY_WEIGHTS%
echo [INFO] Player Model: %PLAYER_WEIGHTS%
echo [INFO] Candy Confidence: %CONFIDENCE%
echo [INFO] Player Confidence: %PLAYER_CONFIDENCE%
echo [INFO] Candy Frame Skip: %CANDY_FRAME_SKIP% (0 = max speed)
echo [INFO] Player Frame Skip: %PLAYER_FRAME_SKIP%

python run_enhanced_bot.py ^
    --weights "%CANDY_WEIGHTS%" ^
    --player-weights "%PLAYER_WEIGHTS%" ^
    --conf %CONFIDENCE% ^
    --player-conf %PLAYER_CONFIDENCE% ^
    --skip %CANDY_FRAME_SKIP% ^
    --player-skip %PLAYER_FRAME_SKIP% ^
    --no-preprocess ^
    --show

echo [INFO] Bot stopped.
pause
