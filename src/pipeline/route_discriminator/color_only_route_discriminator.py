import cv2
import numpy as np
from PIL import ImageDraw

from ..interfaces.data_models import Route
from ..interfaces.errors import BatchAlignmentError


def _kmeans(
    features: np.ndarray, n_clusters: int, random_state: int | None
) -> np.ndarray:
    """Return deterministic k-means labels without an additional dependency."""
    rng = np.random.default_rng(random_state)
    centers = features[rng.choice(len(features), n_clusters, replace=False)]
    for _ in range(100):
        labels = ((features[:, None] - centers) ** 2).sum(2).argmin(1)
        updated = np.array(
            [
                features[labels == index].mean(0)
                if np.any(labels == index)
                else centers[index]
                for index in range(n_clusters)
            ]
        )
        if np.allclose(updated, centers):
            break
        centers = updated
    return labels


class ColorOnlyRouteDiscriminator:
    implementation_id = "color_only_discriminator"

    def __init__(
        self,
        n_clusters: int = 6,
        clustering_method="kmeans",
        random_state: int | None = 0,
        color_space="rgb",
        precomputed_features=None,
        **config,
    ):
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be positive.")
        if clustering_method != "kmeans":
            raise ValueError("Only kmeans is supported.")
        if color_space not in {"rgb", "lab"}:
            raise ValueError("color_space must be 'rgb' or 'lab'.")
        self.n_clusters, self.clustering_method, self.color_space, self.random_state = (
            n_clusters,
            clustering_method,
            color_space,
            random_state,
        )
        self.precomputed_features = dict(precomputed_features or {})

    @property
    def configuration(self):
        """ " color_space must be configured 'rgb' or 'lab' to use either of them."""
        return {
            "n_clusters": self.n_clusters,
            "clustering_method": self.clustering_method,
            "color_space": self.color_space,
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
            features = self.precomputed_features.get(id(image))
            if features is None:
                features = self._colors(image, batch)
            if len(features) != len(batch):
                raise ValueError("precomputed colour count does not match holds.")
            k = min(self.n_clusters, len(batch))
            labels = _kmeans(features, k, self.random_state)
            groups = {}
            for h, label in zip(batch, labels):
                groups.setdefault(int(label), set()).add(h)
            result.append([Route(v, i) for i, v in enumerate(groups.values())])
        return result

    def _colors(self, image, holds):
        array = np.asarray(image.convert("RGB"))
        if self.color_space == "lab":
            array = cv2.cvtColor(array, cv2.COLOR_RGB2LAB)
        result = []
        for hold in holds:
            mask = np.zeros(array.shape[:2], np.uint8)
            cv2.fillPoly(
                mask,
                [np.array([(p.x, p.y) for p in hold.polygon.points], np.int32)],
                1,
            )
            values = array[mask.astype(bool)]
            result.append(values.mean(0) if len(values) else np.zeros(3))
        return np.asarray(result)

    def _color(self, image, hold):
        a = np.asarray(image.convert("RGB"))
        if self.color_space == "lab":
            a = cv2.cvtColor(a, cv2.COLOR_RGB2LAB)
        m = np.zeros(a.shape[:2], np.uint8)
        cv2.fillPoly(
            m, [np.array([(p.x, p.y) for p in hold.polygon.points], np.int32)], 1
        )
        return a[m.astype(bool)].mean(0)

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
