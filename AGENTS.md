write comments in english.

Read docs/architecture.md

When reading logs, please limit the amount read to 100 lines or less.

## Screenshot API
You can take a screenshot of the game by calling the following API:
`GET http://127.0.0.1:8081/screenshot`

The screenshot will be saved to `./tmp/screenshot_<timestamp>.png` and the path will be returned in the JSON response.

### Headless Support
This API supports taking screenshots even in headless mode (using Godot's internal viewport capture). If the internal capture fails, it will fall back to a standard screen grab.



### Slay the Spire 2 - AutoSlayer Utility Classes & Handlers

Below is a list of useful classes and interfaces used by the `AutoSlayer` system. You can leverage these to build a robust and stable AI agent without relying on fragile pixel-based image recognition or manual mouse-click simulations.

#### 1. Core Helpers

* **`WaitHelper` (`MegaCrit.Sts2.Core.AutoSlay.Helpers.WaitHelper`)**
* **Description:** Provides asynchronous polling mechanisms to safely wait for game states to resolve. Instead of hardcoded `Thread.Sleep()`, it waits until internal game action queues are empty, animations are finished, or UI elements become interactable. Using this is critical to prevent the AI from issuing commands before the game is ready.


* **`UiHelper` (`MegaCrit.Sts2.Core.AutoSlay.Helpers.UiHelper`)**
* **Description:** A utility class to directly fetch and manipulate UI components within the Godot node tree. It allows the AI to bypass mouse input simulation and programmatically invoke button clicks or select UI elements (e.g., card rewards, map nodes, shop items).


* **`Watchdog` (`MegaCrit.Sts2.Core.AutoSlay.Helpers.Watchdog`)**
* **Description:** A fail-safe timer mechanism. It monitors the AI's execution flow and throws an `AutoSlayTimeoutException` if the game state hasn't changed for a certain amount of time. This is essential for preventing the AI from getting permanently stuck in unexpected edge cases or pop-ups during unsupervised training/play.



#### 2. Decision Making & Logic

* **`AutoSlayCardSelector` (`MegaCrit.Sts2.Core.AutoSlay.Helpers.AutoSlayCardSelector`)**
* **Description:** Handles the logic for filtering playable cards from the hand based on the current energy/mana. It also evaluates valid targets (e.g., alive enemies) for cards that require a specific target. You can replace or override the logic in this class to integrate your own reinforcement learning (RL) model's predictions.



#### 3. State Machine Handlers (Screen & Room)

The AutoSlayer uses a state-machine pattern. Execution is delegated to specific Handlers based on the current context, ensuring the AI only performs valid actions for the current screen.

* **`IRoomHandler` / `IScreenHandler` (`MegaCrit.Sts2.Core.AutoSlay.Handlers.*`)**
* **Description:** Base interfaces for context-specific handlers.


* **`CombatRoomHandler`**
* **Description:** Manages the battle loop. It checks if it's the player's turn, queries the `AutoSlayCardSelector` for the next move, pushes the corresponding `PlayCardAction` to the game's internal command queue, and signals the `EndPlayerTurnAction`.


* **`MapScreenHandler` / `RewardsScreenHandler` / `EventRoomHandler**`
* **Description:** Dedicated handlers to navigate the map, automatically pick up loot/cards after a battle, or choose random options in text-based events.



#### 4. Action Execution

* **`ActionExecutor` / Command Classes (`MegaCrit.Sts2.Core.GameActions.*`)**
* **Description:** Rather than simulating dragging a card, the AutoSlayer instantiates classes like `PlayCardAction` or `EndPlayerTurnAction` and pushes them directly into the game's internal queue (`ActionExecutor`). This ensures 100% execution accuracy regardless of screen resolution or framerate drops.


*"When implementing the new AI logic, please strictly follow the state-machine pattern using the `IScreenHandler` and `IRoomHandler` interfaces. Always use `WaitHelper` before issuing the next command to ensure the game engine has finished processing the previous action. For UI interactions, use `UiHelper` and emit internal game actions rather than trying to simulate hardware mouse inputs. If you need to write the card selection logic, integrate it within or alongside `AutoSlayCardSelector`."*