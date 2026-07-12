from __future__ import annotations

import time
from contextlib import contextmanager


MINIMUM_DISPLAY_SECONDS = 2.0


def show_water_cooling_loader(placeholder: object) -> None:
    """Show a small CSS scene while the LLM request is running."""
    placeholder.markdown(
        """
        <style>
        .cooling-loader {
            position: relative;
            width: 100%;
            max-width: 135px;
            min-height: 58px;
            aspect-ratio: 1.45 / 1;
            margin: 0.35rem auto 0;
            overflow: hidden;
            border: 1px solid #334155;
            border-radius: 8px;
            background: #111827;
        }
        .cooling-label {
            position: absolute;
            top: 5%;
            left: 6%;
            z-index: 5;
            color: #bfdbfe;
            font-size: 8px;
            font-weight: 600;
        }
        .lake {
            position: absolute;
            left: 4%;
            bottom: -8%;
            width: 42%;
            height: 29%;
            overflow: hidden;
            border-radius: 50%;
            background: #1677b8;
            box-shadow: inset 0 4px 0 #38bdf8;
        }
        .lake::after {
            content: "";
            position: absolute;
            top: 28%;
            left: -15%;
            width: 130%;
            height: 15%;
            border-top: 2px solid rgba(224, 242, 254, 0.72);
            border-radius: 50%;
            animation: lake-wave 1.1s ease-in-out infinite alternate;
        }
        .person {
            position: absolute;
            left: 40%;
            bottom: 10%;
            width: 22%;
            height: 55%;
            animation: pump-bob 0.55s ease-in-out infinite alternate;
        }
        .head {
            position: absolute;
            top: 0;
            left: 28%;
            width: 38%;
            aspect-ratio: 1;
            border: 2px solid #fbbf24;
            border-radius: 50%;
        }
        .body {
            position: absolute;
            top: 25%;
            left: 45%;
            width: 11%;
            height: 42%;
            border-radius: 3px;
            background: #fbbf24;
        }
        .arm, .leg {
            position: absolute;
            width: 10%;
            border-radius: 3px;
            background: #fbbf24;
            transform-origin: top center;
        }
        .arm { top: 30%; left: 50%; height: 34%; transform: rotate(-58deg); }
        .leg.one { top: 64%; left: 50%; height: 34%; transform: rotate(25deg); }
        .leg.two { top: 64%; left: 50%; height: 34%; transform: rotate(-25deg); }
        .hose-inlet {
            position: absolute;
            left: 20%;
            bottom: 16%;
            width: 29%;
            height: 25%;
            z-index: 2;
            border: 4px solid #94a3b8;
            border-top-color: transparent;
            border-right-color: transparent;
            border-radius: 0 0 0 28px;
            transform: rotate(-8deg);
            filter: drop-shadow(0 1px 1px rgba(0, 0, 0, 0.45));
        }
        .hose-outlet {
            position: absolute;
            left: 57%;
            bottom: 45%;
            width: 15%;
            height: 5%;
            min-height: 3px;
            z-index: 3;
            border-radius: 999px;
            background: #94a3b8;
            box-shadow: 0 1px 1px rgba(0, 0, 0, 0.45);
            transform: rotate(5deg);
        }
        .water-jet {
            position: absolute;
            left: 69%;
            bottom: 44%;
            width: 12%;
            height: 5%;
            min-height: 3px;
            z-index: 4;
            border-radius: 999px;
            background: linear-gradient(90deg, #0ea5e9, #7dd3fc);
            box-shadow: 0 0 5px rgba(56, 189, 248, 0.8);
            transform: rotate(5deg);
            transform-origin: left center;
            animation: water-pulse 0.42s ease-in-out infinite alternate;
        }
        .water-jet::before, .water-jet::after {
            content: "";
            position: absolute;
            width: 4px;
            height: 4px;
            border-radius: 50%;
            background: #7dd3fc;
            animation: water-drop 0.55s ease-in-out infinite;
        }
        .water-jet::before { right: 12%; top: -4px; }
        .water-jet::after { right: -8%; top: 3px; animation-delay: 0.2s; }
        .token-cooler {
            position: absolute;
            right: 4%;
            bottom: 12%;
            width: 21%;
            height: 34%;
            border: 1px solid #475569;
            border-radius: 5px;
            background: #0f172a;
        }
        .token-cooler::before {
            content: "AI";
            position: absolute;
            inset: 20%;
            display: grid;
            place-items: center;
            border: 1px solid #60a5fa;
            border-radius: 50%;
            color: #93c5fd;
            font-size: 7px;
            font-weight: 700;
            animation: cooler-pulse 0.7s ease-in-out infinite alternate;
        }
        @keyframes lake-wave { to { transform: translateX(8px) translateY(2px); } }
        @keyframes pump-bob { to { transform: translateY(2px) rotate(1deg); } }
        @keyframes water-pulse { to { transform: rotate(5deg) scaleX(0.82); opacity: 0.7; } }
        @keyframes water-drop { to { transform: translateX(4px) translateY(2px); opacity: 0.25; } }
        @keyframes cooler-pulse { to { box-shadow: 0 0 8px #3b82f6; } }
        </style>
        <div class="cooling-loader" role="status" aria-label="Cooling AI tokens with Bodensee water">
          <div class="cooling-label">Bodensee cooling...</div>
          <div class="lake"></div>
          <div class="hose-inlet"></div>
          <div class="person">
            <div class="head"></div><div class="body"></div><div class="arm"></div>
            <div class="leg one"></div><div class="leg two"></div>
          </div>
          <div class="hose-outlet"></div><div class="water-jet"></div>
          <div class="token-cooler"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@contextmanager
def water_cooling_loader(placeholder: object):
    """Keep the animation visible briefly and remove it after the request."""
    started_at = time.monotonic()
    show_water_cooling_loader(placeholder)
    try:
        yield
    finally:
        remaining = MINIMUM_DISPLAY_SECONDS - (time.monotonic() - started_at)
        if remaining > 0:
            time.sleep(remaining)
        placeholder.empty()
