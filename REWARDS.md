# Clash Royale Bot Reward System

This document outlines all the reward mechanisms implemented in the Clash Royale bot to guide its learning and decision-making process.

## Overview

The bot uses a multi-faceted reward system that encourages strategic gameplay, efficient resource management, and successful tower destruction while penalizing poor defensive play and wasteful actions.

## Core Reward Components

### 1. Enemy Presence Penalty
- **Purpose**: Discourage allowing enemies to accumulate on the battlefield
- **Formula**: `-enemy_presence * 0.5`
- **Calculation**: Sum of absolute positions (x + y) for all detected enemy units
- **Impact**: Continuous pressure to eliminate enemy threats

### 2. Elixir Management Rewards

#### High Elixir Penalty
- **Trigger**: When elixir ≥ 8
- **Penalty**: `-2.0`
- **Purpose**: Prevent elixir waste and encourage active play

#### Elixir Spending Reward
- **Base Reward**: `+elixir_spent * 1.0`
- **Purpose**: Encourage any card play over inaction

#### Effective Elixir Usage Bonus
- **Trigger**: When elixir spending coincides with enemy reduction
- **Bonus**: `+3.0 * min(elixir_spent, enemy_reduced)`
- **Purpose**: Reward strategic plays that effectively counter enemies

### 3. Tower Damage Rewards

#### Enemy Tower Damage (Positive Rewards)
- **Base Reward**: `damage_dealt * 100`
- **Push Bonus**: Additional `damage_dealt * 150` when allies are in enemy territory AND damage > 10% to discourage hitting Towers with spells
- **Destruction Bonus**: 
  - King Tower: `+50`
  - Princess Tower: `+30`

#### Ally Tower Damage (Penalties)
- **Base Penalty**: `-damage_taken * 100`
- **Push Penalty**: Additional `-damage_taken * 150` when enemies are in ally territory AND damage > 10%
- **Destruction Penalty**:
  - King Tower: `-60`
  - Princess Tower: `-40`

### 4. Spell Penalty System
- **Trigger**: When playing spell cards that don't hit any enemies
- **Penalty**: `-5`
- **Detection**: 100-pixel radius around spell target location
- **Purpose**: Discourage wasteful spell usage

### 5. End Game Rewards
- **Victory**: `+100`
- **Defeat**: `-100`
- **Applied**: At episode termination

## Territory-Based Modifiers

### Enemy Territory (Upper Half of Screen, y < 0.5)
- **Effect**: Triggers push bonuses for tower damage
- **Multiplier**: 1.5x additional reward for significant tower damage
- **Purpose**: Reward successful offensive pushes

### Ally Territory (Lower Half of Screen, y > 0.5)
- **Effect**: Triggers push penalties for tower damage taken
- **Multiplier**: 1.5x additional penalty for significant tower damage
- **Purpose**: Heavily penalize poor defensive play

## Special Mechanics

### King Tower Awakening Protection
- **Issue**: King towers show level initially, then HP when awakened
- **Solution**: Detect awakening events and prevent false damage calculations
- **Threshold**: HP reading within 100 of max HP considered awakening
- **Effect**: Prevents massive false rewards/penalties

### Action Masking
- **Valid Actions**: Only actions with sufficient elixir for the card
- **Always Valid**: No-op action
- **Purpose**: Prevent impossible actions

### Unknown Card Handling
- **Trigger**: When all cards are detected as "Unknown"
- **Action**: Click center screen and return 0 reward
- **Purpose**: Graceful handling of detection failures

## Reward Calculation Flow

1. **Base Rewards**: Enemy presence penalty + elixir management
2. **Tower Damage**: Calculate damage-based rewards/penalties with territory modifiers
3. **Spell Penalties**: Apply if spell missed enemies
4. **End Game**: Apply victory/defeat bonuses if game ended
5. **Total**: Sum all components for final reward

## Configuration Constants

```python
# Damage thresholds
SIGNIFICANT_DAMAGE_THRESHOLD = 0.1  # 10% of max HP
SPELL_HIT_RADIUS = 100  # pixels

# Territory boundaries
ENEMY_TERRITORY = y < 0.5  # Upper half
ALLY_TERRITORY = y > 0.5   # Lower half

# Multipliers
BASE_TOWER_MULTIPLIER = 100
PUSH_BONUS_MULTIPLIER = 150
ELIXIR_EFFICIENCY_MULTIPLIER = 3.0
ENEMY_PRESENCE_FACTOR = 0.5
```
