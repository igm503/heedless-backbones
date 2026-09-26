from django.core.exceptions import ValidationError


def choices(scope):
    """Built-in enum values followed by categories approved with approve_category."""
    from .models import Category, PretrainMethod, TokenMixer

    enum = {"model_type": TokenMixer, "pretrain_method": PretrainMethod}[scope]
    values = [item.value for item in enum]
    approved = Category.objects.filter(scope=scope).order_by("name").values_list("name", flat=True)
    values.extend(name for name in approved if name not in values)
    return [(value, value) for value in values]


def validate_category(scope, value):
    if value not in dict(choices(scope)):
        raise ValidationError({scope: f"{value!r} is not an approved category; see approve_category."})
