import os
import json
import inspect
import requests
from PIL import Image
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt


class ChartImgClient:
    def __init__(self, api_key=None):
        self.base_url = (
            "https://api.chart-img.com/v2/tradingview/advanced-chart/storage"
        )
        self.headers = {
            "x-api-key": api_key,
            "content-Type": "application/json",
        }
        self.root_path = Path(__file__).parent.resolve() / "model_cache/images"
        self.root_path.mkdir(exist_ok=True, parents=True)

    def get_chart(
        self,
        symbol,
        interval="1d",
        studies=None,
        override=None,
        width=800,
        height=600,
        chart_style="candle",
    ):
        """
        Fetches a v2 advanced chart from Chart-Img API

        Parameters:
        - symbol (str): Trading symbol (e.g., "NASDAQ:AAPL")
        - interval (str): Time interval (e.g., "1m", "1h", "1d", "4h")
        - studies (list): List of study objects (e.g., [{"name": "RSI"}, {"name": "MACD"}])
        - override (dict): Custom style overrides (e.g., {"style": {"candleStyle.upColor": "green"}})
        - width (int): Image width in pixels
        - height (int): Image height in pixels

        Returns:
        - PIL Image object or None if request fails
        """
        called_from_agent = False
        payload = {
            "symbol": symbol,
            "interval": interval,
            "width": min(width, 800),
            "height": min(height, 600),
            "style": chart_style,
        }

        if studies:
            payload["studies"] = studies

        if override:
            payload["override"] = override

        try:
            response = requests.post(
                self.base_url, headers=self.headers, json=payload, timeout=15
            )

            response.raise_for_status()
            response_url = json.loads(response.text)["url"]

            frame_stack = inspect.stack()
            frame_index = next(
                (
                    i
                    for i, stack in enumerate(frame_stack)
                    if stack.function == "get_chart_img"
                ),
                None,
            )
            content = requests.get(response_url).content
            image = Image.open(BytesIO(content))
            self.save_chart(
                image, self.root_path / f"{symbol.replace(':', '_').lower()}.png"
            )

            if frame_index is not None:
                called_from_agent = (
                    frame_stack[frame_index + 1].function == "finance_agent"
                )

            return response_url if called_from_agent else image

        except requests.exceptions.RequestException as e:
            print(f"Error fetching chart: {e}")
            return None

    def display_chart(self, image):
        """Display the chart using matplotlib"""
        if image:
            plt.figure(figsize=(12, 6))
            plt.imshow(image)
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print("No image to display")

    def save_chart(self, image, filename="v2_chart.png"):
        """Save the chart to a file"""
        if image:
            image.save(self.root_path / filename)
            print(f"Chart saved as {filename}")
        else:
            print("No image to save")


def get_chart_img(
    symbol="MSFT",
    interval="4h",
    chart_style="candle",
    studies=None,
    override=None,
):
    API_KEY = os.getenv("CHART_IMG_API_KEY")
    chart_client = ChartImgClient(api_key=API_KEY)

    # Define v2 chart specifications
    if not studies:
        studies = [
            {"name": "Volume"},
            {"name": "Relative Strength Index"},
            {
                "name": "Directional Movement",
                "override": {
                    "+DI.visible": True,
                    "+DI.linewidth": 1,
                    "+DI.plottype": "line",
                    "+DI.color": "rgb(33,150,243)",
                    "-DI.visible": True,
                    "-DI.linewidth": 1,
                    "-DI.plottype": "line",
                    "-DI.color": "rgb(255,109,0)",
                    "DX.visible": False,
                    "DX.linewidth": 1,
                    "DX.plottype": "line",
                    "DX.color": "rgba(255,255,255,0)",
                    "ADX.visible": True,
                    "ADX.linewidth": 2,
                    "ADX.plottype": "line",
                    "ADX.color": "rgb(245,0,87)",
                    "ADXR.visible": False,
                    "ADXR.linewidth": 1,
                    "ADXR.plottype": "line",
                    "ADXR.color": "rgba(255,255,255,0)",
                },
            },
        ]

    if not override:
        override = {
            "style": {
                "candleStyle.upColor": "rgb(0,255,0)",
                "candleStyle.downColor": "rgb(255,0,0)",
            },
            "showSymbolWatermark": False,
        }

    chart_image = chart_client.get_chart(
        symbol=f"NASDAQ:{symbol}",
        interval=interval,
        studies=studies,
        override=override,
        width=1200,
        height=900,
        chart_style=chart_style,
    )

    # chart_client.display_chart(chart_image)
    # chart_client.save_chart(chart_image, "msft_chart.png")

    return chart_image


if __name__ == "__main__":
    get_chart_img()
