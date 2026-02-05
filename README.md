# Mobile Robotik - Pfadplanung
## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Ausführung

### 2D Pfadplanung (omnidirektionaler Roboter)

```bash
python main.py
```

Testet alle Kombinationen aus:
- **Maps:** easy, medium, hard
- **Roboter:** circle, rectangle, triangle
- **Algorithmen:** A*, Dijkstra, Best-First, RRT, PRM

### 3D Pfadplanung (mit Orientierung)

```bash
python main_3d.py
```
