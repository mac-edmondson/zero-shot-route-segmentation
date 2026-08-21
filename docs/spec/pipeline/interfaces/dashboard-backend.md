# Dashboard Backend API

**Depends on:** [RouteDiscriminatorPipeline](route-discriminator-pipeline.md), [EvaluationSuite](evaluation-suite.md), optionally [AugmentationSuite](augmentation-suite.md)
**Consumed by:** [Dashboard Frontend](dashboard-frontend.md)

**Note: While this is called a spec, what is defined in this document is not final and may not be implemented exactly as described. This is more of a rough outline defined for initial collaboration.***

## 1. Responsibility

Expose the route-discriminator pipeline and evaluation suite through a stable application API for the working demo.

## 2. Specification Overview Diagram

![API Spec. Diagram](/docs/diagrams/spec_rest_api.drawio.svg)

## 3. REST API Endpoint Contract

Note: When a browser connects to the API, an "HTTP Only" cookie is installed which acts as a session identifier. All API endpoints use this session identifier such that endpoints look like a singleton, but are actually per individually connected browser session.

For long-running operations, only one operation of a given type may be active per session. If an operation is started while another operation of the same type is still processing, the API returns `409 Conflict`.

### Pipeline Configuration

These endpoints are used to configure the inference pipeline for the session.

#### `GET /pipeline/available_configs`

Get a list of available pipeline configurations.

##### Request body

None specified.

##### Response

###### Status Code(s): `200 OK`

###### Body

* `hold_detector`: list[string]
* `route_classifier`: list[string]

#### `PUT /pipeline`

Replace the existing pipeline object in the backend according to the components specified in the request body.

##### Request body

* `hold_detector`: string (value provided from `available_configs`)
* `route_classifier`: string (value provided from `available_configs`)

##### Response

###### Status Code(s): `200 OK`

###### Body

None specified.

---

#### `GET /pipeline`

Returns the current pipeline configuration.

##### Request body

None specified.

##### Response

###### Status Code(s): `200 OK`

###### Body

* `hold_detector`: string (allowed values as listed in `available_configs`)
* `route_classifier`: string (allowed values as listed in `available_configs`)

---

### Images / Gallery

**Endpoint root:** `/images`

No code needed for implementation of this endpoint. For specification of this endpoint, see [Nginx Autoindex](https://nginx.org/en/docs/http/ngx_http_autoindex_module.html)

---

### Image Augmentation

#### `PUT /image/working`

Sets / replaces the current working image for the session.

##### Request body

Contains:

* `image`: image.

##### Response

###### Status Code(s): `200 OK`

###### Body

None specified.

---

#### `GET /image/working`

Returns the state of the current working image.

The frontend can poll this endpoint after kicking off any long-running task which returns early (e.g. `augment`) until `status` becomes `"completed"` or `"failed"`.

##### Request body

None specified.

##### Response

###### Status Code(s): `200 OK`

###### Body

* `status`: `"processing"`, `"completed"`, or `"failed"`.
* `image`: image.
* `error`: error information when `status` is `"failed"`; otherwise omitted.

---

#### `POST /image/working/segment`

Send a request to segment the image with SAM2 on the backend based on given coordinates.

##### Request body

Contains:

* `coordinates`: list of coordinates where the user clicked on the frontend to indicate objects to segment with SAM.

##### Response

###### Status Code(s): `202 Accepted`, `409 Conflict`

Returns immediately after the segmentation request is accepted. The frontend must poll the corresponding GET endpoint to retrieve the segments.

###### Body

None specified.

#### `GET /image/working/segment`

##### Request body

None specified.

##### Response

###### Status Code(s): `200 OK`

###### Body

* `status`: `"processing"`, `"completed"`, or `"failed"`.
* `segments`: list of segment objects.

  * `id`: a `segment_id`.
  * `polygon`: a polygon.
* `error`: error information when `status` is `"failed"`; otherwise omitted.

---

#### `POST /image/working/augment`

Starts augmentation of segments in the working image.

##### Request body

A list of augmentation objects, where each object contains:

* `segment_id`: id (from the segment `id`s provided by the backend).
* `chalk_percent`: float.
* `color`: RGB color.

##### Response

###### Status Code(s): `202 Accepted`, `409 Conflict`

Returns immediately after the augmentation request is accepted. For the augmented image, the frontend should poll the `GET /image/working` endpoint.

###### Body

None specified.

---

#### `DELETE /image/working/segment/{segment_id}`

Deletes a segment from the working image.

#### Path parameter

* `segment_id`: id of the segment to delete.

#### Request body

None specified.

#### Response

###### Status Code(s): `204 No Content`

###### Body

None specified.

---

#### `POST /pipeline/infer/working`

Starts inference on the current working image.

#### Request body

None specified.

#### Response

###### Status Code(s): `202 Accepted`, `409 Conflict`

Returns immediately after the inference request is accepted. The frontend must poll the corresponding GET endpoint to retrieve the inferred routes.

###### Body

None specified.

---

### `GET /pipeline/infer/working`

Returns the current inference state and result.

#### Request body

None specified.

#### Response

###### Status Code(s): `200 OK`

The frontend can poll this endpoint until `status` becomes `"completed"` or `"failed"`.

###### Body

* `routes`: list of routes, where each route is itself a list of holds.
* `status`: `"processing"`, `"completed"`, or `"failed"`.
* `error`: error information when `status` is `"failed"`; otherwise omitted.
