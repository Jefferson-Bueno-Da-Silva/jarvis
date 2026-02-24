from __future__ import annotations

from typing import Any

from langchain.tools import tool
from pydantic import BaseModel, Field

from googleTask import GoogleTask


_google_tasks_client: GoogleTask | None = None


def _get_google_tasks_client() -> GoogleTask:
    global _google_tasks_client
    if _google_tasks_client is None:
        _google_tasks_client = GoogleTask()
    return _google_tasks_client


def _serialize_task(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": task.get("id"),
        "title": task.get("title"),
        "status": task.get("status"),
        "due": task.get("due"),
        "notes": task.get("notes"),
        "updated": task.get("updated"),
    }


class ListTasksInput(BaseModel):
    limit: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum number of tasks returned (1 to 100).",
    )
@tool(
    "google_tasks_list",
    args_schema=ListTasksInput,
    description="List tasks from the default Google Tasks list.",
)
def google_tasks_list(limit: int = 20) -> dict[str, Any]:
    """Return task list in a predictable structure for agent consumption."""
    try:
        tasks = _get_google_tasks_client().getTasks() or []
        serialized = [_serialize_task(task) for task in tasks[:limit]]
        return {
            "ok": True,
            "count": len(serialized),
            "tasks": serialized,
            "message": "Tasks listed successfully.",
        }
    except Exception as error:
        return {"ok": False, "error": f"Failed to list tasks: {error}"}


class CreateTaskInput(BaseModel):
    title: str = Field(..., min_length=1, description="Task title.")
    notes: str | None = Field(default=None, description="Optional task notes.")
    due: str | None = Field(
        default=None,
        description="Optional due date in RFC3339 format (e.g. 2026-02-25T12:00:00.000Z).",
    )
@tool(
    "google_tasks_create",
    args_schema=CreateTaskInput,
    description="Create a task in Google Tasks default list.",
)
def google_tasks_create(title: str, notes: str | None = None, due: str | None = None) -> dict[str, Any]:
    try:
        created = _get_google_tasks_client().createTask(title=title, notes=notes, due=due)
        if created is None:
            return {"ok": False, "error": "Task could not be created."}
        return {"ok": True, "task": _serialize_task(created), "message": "Task created successfully."}
    except Exception as error:
        return {"ok": False, "error": f"Failed to create task: {error}"}


class UpdateTaskInput(BaseModel):
    task_id: str = Field(..., min_length=1, description="Task ID to update.")
    title: str | None = Field(default=None, description="New task title.")
    notes: str | None = Field(default=None, description="New task notes.")
    due: str | None = Field(
        default=None,
        description="New due date in RFC3339 format (e.g. 2026-02-25T12:00:00.000Z).",
    )
    status: str | None = Field(
        default=None,
        description="Task status. Valid values: needsAction or completed.",
    )
@tool(
    "google_tasks_update",
    args_schema=UpdateTaskInput,
    description="Update an existing task by task ID.",
)
def google_tasks_update(
    task_id: str,
    title: str | None = None,
    notes: str | None = None,
    due: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    if status not in (None, "needsAction", "completed"):
        return {
            "ok": False,
            "error": "Invalid status. Use one of: needsAction, completed.",
        }

    if not any(value is not None for value in (title, notes, due, status)):
        return {"ok": False, "error": "At least one field to update must be provided."}

    try:
        updated = _get_google_tasks_client().updateTask(
            task_id=task_id,
            title=title,
            notes=notes,
            due=due,
            status=status,
        )
        if updated is None:
            return {"ok": False, "error": "Task could not be updated."}
        return {"ok": True, "task": _serialize_task(updated), "message": "Task updated successfully."}
    except Exception as error:
        return {"ok": False, "error": f"Failed to update task: {error}"}


class DeleteTaskInput(BaseModel):
    task_id: str = Field(..., min_length=1, description="Task ID to delete.")
@tool(
    "google_tasks_delete",
    args_schema=DeleteTaskInput,
    description="Delete a task by task ID from Google Tasks default list.",
)
def google_tasks_delete(task_id: str) -> dict[str, Any]:
    try:
        deleted = _get_google_tasks_client().deleteTask(task_id=task_id)
        if not deleted:
            return {"ok": False, "error": "Task could not be deleted."}
        return {"ok": True, "task_id": task_id, "message": "Task deleted successfully."}
    except Exception as error:
        return {"ok": False, "error": f"Failed to delete task: {error}"}


GOOGLE_TASKS_TOOLS = [
    google_tasks_list,
    google_tasks_create,
    google_tasks_update,
    google_tasks_delete,
]
