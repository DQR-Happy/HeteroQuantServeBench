"""Local management plane, independent of the selected machine's availability."""

import asyncio

from fastapi import APIRouter, Request

from hqsb.console.machine_registry import MachineConfig, MachineError


def selected_machine(request: Request) -> str:
    return request.headers.get("x-hqsb-machine", "local")


def machine_router(registry, connections):
    router = APIRouter(tags=["machines"])

    @router.get("/machines")
    def listing(request: Request):
        return registry.public(selected_machine(request))

    @router.post("/machines", status_code=201)
    def add(body: MachineConfig):
        registry.add(body)
        return body.model_dump(exclude_none=True)

    @router.post("/machines/{identity}/probe")
    async def probe(identity: str):
        if identity == "local":
            return {
                "status": "ready",
                "machine_id": "local",
                "node_id": registry.node_id,
            }
        return await asyncio.to_thread(connections.probe, registry.get(identity))

    @router.post("/machines/{identity}/activate")
    async def activate(identity: str):
        # Switching is view selection only. Existing requests retain their
        # worker/record owner and continue on their original node.
        if identity != "local":
            await asyncio.to_thread(connections.probe, registry.get(identity))
        return {"selected_id": identity, "reload_required": True}

    @router.post("/machines/{identity}/remove")
    def remove(identity: str, request: Request):
        if identity == "local" or identity == selected_machine(request):
            raise MachineError(
                "MACHINE_IN_USE", "Switch away before removing this connection", 409
            )
        # Keep an established tunnel until shutdown so other browser sessions
        # and already streamed responses are not interrupted.
        registry.remove(identity)
        return {"removed": identity}

    return router
