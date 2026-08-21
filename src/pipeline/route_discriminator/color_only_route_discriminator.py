import cv2
import numpy as np
from PIL import ImageDraw

from ..interfaces.data_models import Route
from ..interfaces.errors import BatchAlignmentError


class ColorOnlyRouteDiscriminator:
    implementation_id = "color_only_discriminator"

    def __init__(
        self,
        n_clusters: int,
        clustering_method="kmeans",
        random_state: int | None = 0,
        **config,
    ):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if clustering_method != "kmeans":
            raise ValueError("Only kmeans is supported.")
        self.n_clusters, self.clustering_method, self.random_state = (
            n_clusters,
            clustering_method,
            random_state,
        )

    @property
    def configuration(self):
        return {
            "n_clusters": self.n_clusters,
            "clustering_method": self.clustering_method,
            "random_state": self.random_state,
        }

    def get_routes(self, images, holds):
        if len(images) != len(holds):
            raise BatchAlignmentError("images and holds must align.")
        result = []
        for image, batch in zip(images, holds):
            if not batch:
                result.append([])
                continue
            features = np.array([self._color(image, h) for h in batch])
            k = min(self.n_clusters, len(batch))
            labels = self._kmeans(features, k)
            groups = {}
            for h, label in zip(batch, labels):
                groups.setdefault(int(label), set()).add(h)
            result.append([Route(v, i) for i, v in enumerate(groups.values())])
        return result

    def _color(self, image, hold):
        a = np.asarray(image.convert("RGB"))
        m = np.zeros(a.shape[:2], np.uint8)
        cv2.fillPoly(
            m, [np.array([(p.x, p.y) for p in hold.polygon.points], np.int32)], 1
        )
        return a[m.astype(bool)].mean(0)

    def _kmeans(self, x, k):
        rng = np.random.default_rng(self.random_state)
        centers = x[rng.choice(len(x), k, replace=False)]
        for _ in range(100):
            labels = ((x[:, None] - centers) ** 2).sum(2).argmin(1)
            new = np.array(
                [
                    x[labels == i].mean(0) if np.any(labels == i) else centers[i]
                    for i in range(k)
                ]
            )
            if np.allclose(new, centers):
                break
            centers = new
        return labels

    @staticmethod
    def mark_routes(images, routes):
        if len(images) != len(routes):
            raise BatchAlignmentError("images and routes must align.")
        out = []
        for image, batch in zip(images, routes):
            overlay = image.convert("RGB").copy()
            d = ImageDraw.Draw(overlay)
            for route in batch:
                color = (
                    (route.route_id * 97) % 255,
                    (route.route_id * 57) % 255,
                    (route.route_id * 37) % 255,
                )
                for h in route.holds:
                    d.line(
                        [(p.x, p.y) for p in (*h.polygon.points, h.polygon.points[0])],
                        fill=color,
                        width=3,
                    )
            out.append(overlay)
        return out
