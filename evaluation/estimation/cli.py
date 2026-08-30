"""Installed CLI shim for six-set pose evaluation."""

from .evaluate import build_parser, evaluate_methods, main

__all__ = ["build_parser", "evaluate_methods", "main"]
